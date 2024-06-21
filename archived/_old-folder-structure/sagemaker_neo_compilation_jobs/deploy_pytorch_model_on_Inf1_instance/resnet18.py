def input_fn(request_body, request_content_type):
    import io

    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    f = io.BytesIO(request_body)
    input_image = Image.open(f).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch
