# ONNX Runtime usa chamadas via JNI nativo — sem este keep, o R8 pode remover
# ou renomear classes referenciadas nativamente e a sessão falha só no release.
-keep class ai.onnxruntime.** { *; }

# CameraX / Camera2 usam reflection em alguns pontos internos.
-dontwarn androidx.camera.**
