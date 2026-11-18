"""Phase 4 (CO-2): learned document segmentation, so page detection works on any background.

``model`` and ``train`` need PyTorch (training only); ``infer`` runs the exported ONNX model
with OpenCV's DNN module, so the app itself never needs PyTorch.
"""
