Step 1 — Reinstall PyTorch nightly cu129 and its dependencies cleanly

#This will reinstall PyTorch + torchvision + torchaudio + cu12 NCCL properly, overwriting anything broken:

pip uninstall torch torchvision torchaudio -y
pip install --index-url https://download.pytorch.org/whl/nightly/cu129 torch torchvision torchaudio --force-reinstall


Step 2 — Check that PyTorch sees the correct NCCL
ldd $(python -c "import torch; print(torch.__file__.replace('__init__.py','lib/libtorch_cuda.so'))") | grep nccl


You should see:
libnccl.so.2 => /home/jking/workspace/intern/lib/python3.10/site-packages/torch/lib/nvidia/nccl/libnccl.so.2

No references to cu11 or system /usr/lib paths.


1.Uninstall everything PyTorch/NVIDIA related:
pip uninstall torch torchvision torchaudio nvidia-nccl-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 -y

2.Reinstall nightly cu129 PyTorch:
pip install --index-url https://download.pytorch.org/whl/nightly/cu129 torch torchvision torchaudio --force-reinstall

3.Verify correct NCCL linkage
ldd $(python -c "import torch; print(torch.__file__.replace('__init__.py','lib/libtorch_cuda.so'))") | grep nccl

Should show something like:
libnccl.so.2 => /home/jking/workspace/intern/lib/python3.10/site-packages/torch/lib/nvidia/nccl/libnccl.so.2
No cu11 libraries, no system libraries.


4.Fix the LD_PRELOAD issue:
#Check your current LD_PRELOAD:
echo $LD_PRELOAD
If it shows:
/home/jking/workspace/intern/lib/python3.10/site-packages/torch/lib/nvidia/nccl/libnccl.so.2
Unset it:

unset LD_PRELOAD

Also make sure LD_LIBRARY_PATH isn’t pointing to old libraries:

unset LD_LIBRARY_PATH

Test PyTorch
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.version.nccl); print(torch.cuda.is_available())"


You should see:
torch nightly version (2.10.0.dev...)
CUDA 12.9
NCCL 2.27.5
True for GPU availability

✅ 4️⃣ Run your fine-tuning script
python finetune_internvl.py


It should now work without undefined symbol: ncclMemFree or cannot open shared object file errors.
