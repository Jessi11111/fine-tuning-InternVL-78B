# fine-tuning-InternVL-78B
 NVIDIA RTX PRO 6000 Blackwell, 96 GB of VRAM * 2

 What we need:
┌──────────────────────────────┐
│  Pretrained Model Weights    │  (InternVL3-78B files)
└──────────────────────────────┘
┌──────────────────────────────┐
│      Transformers Library     │  (model architecture + loader)
└──────────────────────────────┘
┌──────────────────────────────┐
│           PyTorch            │  (tensor ops, CUDA, math)
└──────────────────────────────┘
PyTorch = the main underlying library (foundation)
Transformers = a separate library built on top of PyTorch
Pretrained models = files loaded by Transformers and executed by PyTorch





 1.create a new venv:
 python -m venv intern(python3 -m venv intern)
 source intern/bin/activate


2. install torch in Venv:
"""
notes:The  NVIDIA RTX PRO 6000 Blackwell GPU uses the sm_120 compute capability and is already supported in PyTorch 2.7.0+ with CUDA 12.8+

python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_properties(0)); print(torch.cuda.is_available()); print(torch.randn(1).cuda())"
2.7.1+cu128
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120']
_CudaDeviceProperties(name='NVIDIA RTX PRO 6000 Blackwell Server Edition', major=12, minor=0, total_memory=97250MB, multi_processor_count=188, uuid=..., L2_cache_size=128MB)
True
tensor([-0.7635], device='cuda:0')
"""


#uninstall existing PyTorch packages
pip uninstall torch torchvision torchaudio -y

#installs a nightly build of PyTorch with CUDA 12.8, which supports sm_120 (RTX 5090, RTX 6000)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

#to check:
#torch.__version__ → nightly version
#torch.version.cuda → 12.8
#torch.cuda.is_available() → True
#torch.cuda.get_device_name(0) → NVIDIA GeForce RTX 6000
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"




3.other packages in venv:
#torch — already installed
#pandas — CSV handling
#transformers — HuggingFace (InternVL support)
#Pillow — image processing (PIL)
#numpy — numerical operations
#Optional (recommended):
#accelerate — better GPU memory management

pip install pandas transformers pillow numpy
pip install accelerate

pip freeze > requirements.txt



4. Load InternVL-78B model with device mapping:

from transformers import AutoModel, AutoTokenizer
model_name = "OpenGVLab/InternVL3-78B-hf"
model = AutoModel.from_pretrained(
    model_name,
    device_map="auto",             # auto GPU/CPU assignment
    torch_dtype="bfloat16",        # Blackwell loves BF16
    load_in_4bit=True,             # QLoRA
    offload_folder="/mnt/offload", # optional but good for safety
    max_memory={
        0: "94GB",                 # leave a little headroom
        1: "94GB",
        "cpu": "300GB"
    }
)

tokenizer = AutoTokenizer.from_pretrained(model_name)


