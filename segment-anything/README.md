# Segment Anything Model

This is an example deploying Segment Anything Model (SAM) with truss weights preloaded

## Deploy to Baseten
To deploy the model, run the following from the root of the directory

```
truss push --publish
```

## Predict
Example prediction:

```
truss predict --published -d '{"image_url": "https://as2.ftcdn.net/v2/jpg/00/66/26/87/1000_F_66268784_jccdcfdpf2vmq5X8raYA8JQT0sziZ1H9.jpg"}'
```
