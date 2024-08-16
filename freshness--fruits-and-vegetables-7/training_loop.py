import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLOv4(num_classes=len(class_names)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = yolo_loss(outputs, labels, num_classes=len(class_names))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
