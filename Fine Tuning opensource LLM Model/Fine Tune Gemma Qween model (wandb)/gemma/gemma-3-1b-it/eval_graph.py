import cv2
import numpy as np
import math

def save_loss_graph_opencv(log_history, save_path="train_val_loss.png", img_size=(800, 600)):
    width, height = img_size
    margin = 60
    ticks = 5  # number of axis ticks

    train_losses, val_losses = [], []
    steps_train, steps_val = [], []

    for log in log_history:
        if "loss" in log:
            train_losses.append(log["loss"])
            steps_train.append(log["step"])
        if "eval_loss" in log:
            val_losses.append(log["eval_loss"])
            steps_val.append(log["step"])

    if not train_losses:
        print("No training loss found!")
        return

    all_losses = train_losses + val_losses
    min_loss, max_loss = min(all_losses), max(all_losses)

    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    def transform(x, y):
        x_img = int(margin + (x - steps_train[0]) / (steps_train[-1] - steps_train[0]) * (width - 2 * margin))
        y_img = int(height - margin - (y - min_loss) / (max_loss - min_loss) * (height - 2 * margin))
        return x_img, y_img

    # Axes
    cv2.line(img, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 2)
    cv2.line(img, (margin, margin), (margin, height - margin), (0, 0, 0), 2)

    # X-axis ticks (steps)
    for i in range(ticks + 1):
        step = steps_train[0] + i * (steps_train[-1] - steps_train[0]) / ticks
        x, y = transform(step, min_loss)
        cv2.line(img, (x, height - margin - 5), (x, height - margin + 5), (0, 0, 0), 1)
        cv2.putText(img, f"{int(step)}", (x - 10, height - margin + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Y-axis ticks (loss)
    for i in range(ticks + 1):
        loss = min_loss + i * (max_loss - min_loss) / ticks
        x, y = transform(steps_train[0], loss)
        cv2.line(img, (margin - 5, y), (margin + 5, y), (0, 0, 0), 1)
        cv2.putText(img, f"{loss:.2f}", (5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Train loss (blue)
    for i in range(1, len(train_losses)):
        cv2.line(img,
                 transform(steps_train[i - 1], train_losses[i - 1]),
                 transform(steps_train[i], train_losses[i]),
                 (255, 0, 0), 2)

    # Validation loss (red)
    for i in range(1, len(val_losses)):
        cv2.line(img,
                 transform(steps_val[i - 1], val_losses[i - 1]),
                 transform(steps_val[i], val_losses[i]),
                 (0, 0, 255), 2)

    cv2.putText(img, "Train Loss (Blue)", (margin, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(img, "Validation Loss (Red)", (margin + 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "Steps", (width // 2 - 30, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Loss", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imwrite(save_path, img)
    print(f"Loss graph saved to {save_path}")




def save_accuracy_graph_opencv(
    log_history,
    save_path="train_val_accuracy.png",
    img_size=(800, 600),
):
    width, height = img_size
    margin = 60
    ticks = 5

    train_acc, val_acc = [], []
    steps_train, steps_val = [], []

    for log in log_history:
        if "loss" in log:
            # Approximate train accuracy from loss
            train_acc.append(math.exp(-log["loss"]))
            steps_train.append(log["step"])

        if "eval_accuracy" in log:
            val_acc.append(log["eval_accuracy"])
            steps_val.append(log["step"])

    if not val_acc:
        print("No accuracy data found!")
        return

    all_acc = train_acc + val_acc
    min_acc, max_acc = min(all_acc), max(all_acc)

    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    def transform(x, y):
        x_img = int(
            margin
            + (x - steps_train[0])
            / (steps_train[-1] - steps_train[0])
            * (width - 2 * margin)
        )
        y_img = int(
            height
            - margin
            - (y - min_acc)
            / (max_acc - min_acc)
            * (height - 2 * margin)
        )
        return x_img, y_img

    # Axes
    cv2.line(img, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 2)
    cv2.line(img, (margin, margin), (margin, height - margin), (0, 0, 0), 2)

    # X-axis ticks (steps)
    for i in range(ticks + 1):
        step = steps_train[0] + i * (steps_train[-1] - steps_train[0]) / ticks
        x, y = transform(step, min_acc)
        cv2.line(img, (x, height - margin - 5), (x, height - margin + 5), (0, 0, 0), 1)
        cv2.putText(img, f"{int(step)}", (x - 10, height - margin + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Y-axis ticks (accuracy)
    for i in range(ticks + 1):
        acc = min_acc + i * (max_acc - min_acc) / ticks
        x, y = transform(steps_train[0], acc)
        cv2.line(img, (margin - 5, y), (margin + 5, y), (0, 0, 0), 1)
        cv2.putText(img, f"{acc:.2f}", (5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Train accuracy (blue)
    for i in range(1, len(train_acc)):
        cv2.line(
            img,
            transform(steps_train[i - 1], train_acc[i - 1]),
            transform(steps_train[i], train_acc[i]),
            (255, 0, 0),
            2,
        )

    # Validation accuracy (green)
    for i in range(1, len(val_acc)):
        cv2.line(
            img,
            transform(steps_val[i - 1], val_acc[i - 1]),
            transform(steps_val[i], val_acc[i]),
            (0, 255, 0),
            2,
        )

    # Labels
    cv2.putText(img, "Train Accuracy (Blue)", (margin, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(img, "Validation Accuracy (Green)", (margin + 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(img, "Steps", (width // 2 - 30, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Accuracy", (10, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imwrite(save_path, img)
    print(f"Train + Validation Accuracy graph saved to {save_path}")
