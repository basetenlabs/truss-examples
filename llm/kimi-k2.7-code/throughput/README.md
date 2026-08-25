Kimi 2.7 code throughput.
Notes:
In order to optimize for performance, several configurations were stripped out from this config, so this config is good if you are only using for single replica, without KV cache aware routing. The following were removed from this config:

- BIS KV routing
- b10_vision_config
- spec decoding (eagle 3)

Please refer to the BIS registry for golden config.