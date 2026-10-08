Flux's repeated transformer blocks are regionally compiled and the resulting
TorchInductor artifacts are published to a versioned shared cache. Pinned model
and serving revisions keep that cache compatible across replica starts.