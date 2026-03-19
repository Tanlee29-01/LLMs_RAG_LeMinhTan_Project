import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_ollama import ChatOllama
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline

DEFAULT_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "llama3.1:8b")
FALLBACK_MODEL_NAME = os.getenv("RAG_FALLBACK_MODEL_NAME", "llama3.1:8b")
DEFAULT_HF_HOME = r"D:\HuggingFace_Cache"
DEFAULT_LOCAL_MODEL_ROOT = r"D:\models"
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


class OllamaModelHandle:
    """Lightweight handle so existing Streamlit wiring can stay unchanged."""

    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url
        self._copilot_model_name = model_name


def _is_ollama_model(model_name: str) -> bool:
    # Ollama tags are typically "name:tag" (e.g. llama3.1:8b), not HF repo ids.
    return ":" in model_name and "/" not in model_name


def _build_ollama_chat_llm(
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    do_sample: bool | None,
):
    # Ollama uses `num_predict` for max output tokens.
    resolved_temperature = temperature if (do_sample if do_sample is not None else temperature > 0) else 0.0
    return ChatOllama(
        model=model_name,
        base_url=DEFAULT_OLLAMA_BASE_URL,
        temperature=resolved_temperature,
        num_predict=max_new_tokens,
        repeat_penalty=repetition_penalty,
    )


def _configure_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(DEFAULT_HF_HOME, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(DEFAULT_HF_HOME, "transformers"))
    if os.name != "nt":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    elif os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip().lower() == "expandable_segments:true":
        # This allocator setting is unsupported on Windows and only produces noisy warnings.
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
    os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "4")


def _build_generation_config(
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    do_sample: bool | None,
    no_repeat_ngram_size: int,
    **generation_kwargs,
) -> GenerationConfig:
    base_config = getattr(model, "generation_config", None)
    if base_config is not None:
        generation_config = GenerationConfig(**base_config.to_dict())
    else:
        generation_config = GenerationConfig.from_model_config(model.config)

    should_sample = do_sample if do_sample is not None else temperature > 0

    generation_config.max_new_tokens = max_new_tokens
    generation_config.pad_token_id = tokenizer.eos_token_id
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id]
    if isinstance(im_end_id, int) and im_end_id != tokenizer.eos_token_id:
        eos_ids.append(im_end_id)
    generation_config.eos_token_id = eos_ids if len(eos_ids) > 1 else eos_ids[0]
    generation_config.repetition_penalty = repetition_penalty
    generation_config.no_repeat_ngram_size = no_repeat_ngram_size
    generation_config.do_sample = should_sample

    if should_sample:
        generation_config.temperature = temperature
    else:
        # Keep greedy decoding clean to avoid warnings about sampling-only flags.
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None

    for key, value in generation_kwargs.items():
        if not hasattr(generation_config, key):
            continue
        if not should_sample and key in {"temperature", "top_p", "top_k"}:
            continue
        setattr(generation_config, key, value)

    return generation_config


def _resolve_local_model_dir(model_name: str, local_model_dir: str | None = None) -> Path:
    if local_model_dir:
        return Path(local_model_dir)

    model_root = Path(os.getenv("LOCAL_MODEL_ROOT", DEFAULT_LOCAL_MODEL_ROOT))
    return model_root / model_name.split("/")[-1]


def _has_complete_snapshot(local_model_dir: Path) -> bool:
    if not local_model_dir.exists() or not any(local_model_dir.iterdir()):
        return False

    shard_files = list(local_model_dir.glob("*.safetensors"))
    if shard_files:
        return True

    index_path = local_model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return False

    try:
        with index_path.open("r", encoding="utf-8") as index_file:
            index_data = json.load(index_file)
    except (OSError, json.JSONDecodeError):
        return False

    weight_map = index_data.get("weight_map", {})
    required_files = {file_name for file_name in weight_map.values() if file_name}
    return bool(required_files) and all((local_model_dir / file_name).exists() for file_name in required_files)


def resolve_effective_model_name(model_name: str = DEFAULT_MODEL_NAME) -> str:
    if _is_ollama_model(model_name):
        return model_name

    requested_dir = _resolve_local_model_dir(model_name)
    if _has_complete_snapshot(requested_dir):
        return model_name

    fallback_dir = _resolve_local_model_dir(FALLBACK_MODEL_NAME)
    if model_name != FALLBACK_MODEL_NAME and _has_complete_snapshot(fallback_dir):
        return FALLBACK_MODEL_NAME

    return model_name


def _ensure_local_model(model_name: str, local_model_dir: Path) -> str:
    if not _has_complete_snapshot(local_model_dir):
        local_model_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_name,
            local_dir=str(local_model_dir),
            local_dir_use_symlinks=False,
            token=os.getenv("HF_TOKEN"),
        )

    return str(local_model_dir)


def _build_quantization_config() -> tuple[BitsAndBytesConfig | None, torch.dtype]:
    if torch.cuda.is_available():
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return quantization_config, compute_dtype

    return None, torch.float32


def get_hf_llm(
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_token: int = 256,
    temperature: float = 0.0,
    device_map: str = "auto",
    local_model_dir: str | None = None,
    repetition_penalty: float = 1.4,
    do_sample: bool | None = None,
    **generation_kwargs,
):
    if _is_ollama_model(model_name):
        return _build_ollama_chat_llm(
            model_name=model_name,
            max_new_tokens=max_new_token,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

    _configure_hf_cache()
    effective_model_name = resolve_effective_model_name(model_name)
    resolved_model_dir = _resolve_local_model_dir(effective_model_name, local_model_dir)
    model_path = _ensure_local_model(effective_model_name, resolved_model_dir)
    quantization_config, torch_dtype = _build_quantization_config()

    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "dtype": torch_dtype,           # fix warning: torch_dtype → dtype
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "trust_remote_code": True,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    generation_config = _build_generation_config(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        no_repeat_ngram_size=3,
        **generation_kwargs,
    )
    model.generation_config = generation_config

    model_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )
    
    pipeline_llm = HuggingFacePipeline(pipeline=model_pipeline)
    return ChatHuggingFace(llm=pipeline_llm)


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    device_map: str = "auto",
    local_model_dir: str | None = None,
):
    """Nạp model 1 lần duy nhất vào VRAM."""
    if _is_ollama_model(model_name):
        effective_model_name = resolve_effective_model_name(model_name)
        return OllamaModelHandle(effective_model_name, DEFAULT_OLLAMA_BASE_URL), None

    _configure_hf_cache()
    effective_model_name = resolve_effective_model_name(model_name)
    resolved_model_dir = _resolve_local_model_dir(effective_model_name, local_model_dir)
    model_path = _ensure_local_model(effective_model_name, resolved_model_dir)
    quantization_config, torch_dtype = _build_quantization_config()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "dtype": torch_dtype,           # fix warning: torch_dtype → dtype
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "trust_remote_code": True,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    model._copilot_model_name = effective_model_name
    return model, tokenizer


def build_hf_llm(
    model,
    tokenizer,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    repetition_penalty: float = 1.3,
    do_sample: bool | None = None,
    no_repeat_ngram_size: int = 3,
) -> HuggingFacePipeline:
    """Tạo pipeline từ model đã load — không tốn thêm VRAM."""
    if isinstance(model, OllamaModelHandle):
        return _build_ollama_chat_llm(
            model_name=model.model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

    generation_config = _build_generation_config(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    model.generation_config = generation_config

    model_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )
    pipeline_llm = HuggingFacePipeline(pipeline=model_pipeline)
    return ChatHuggingFace(llm=pipeline_llm)
