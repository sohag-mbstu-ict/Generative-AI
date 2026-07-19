import cv2
import numpy as np
import math


def save_perplexity_graph_opencv(
    log_history,
    save_path="train_val_perplexity.png",
    img_size=(800, 600),
):
    width, height = img_size
    margin = 60
    ticks = 5

    train_ppl, val_ppl = [], []
    steps_train, steps_val = [], []

    # Extract perplexity
    for log in log_history:
        if "loss" in log:
            train_ppl.append(math.exp(log["loss"]))
            steps_train.append(log["step"])
        if "eval_loss" in log:
            val_ppl.append(math.exp(log["eval_loss"]))
            steps_val.append(log["step"])

    if not train_ppl:
        print("No perplexity data found!")
        return

    all_ppl = train_ppl + val_ppl
    min_ppl, max_ppl = min(all_ppl), max(all_ppl)

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
            - (y - min_ppl)
            / (max_ppl - min_ppl)
            * (height - 2 * margin)
        )
        return x_img, y_img

    # Axes
    cv2.line(img, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 2)
    cv2.line(img, (margin, margin), (margin, height - margin), (0, 0, 0), 2)

    # X-axis ticks (steps)
    for i in range(ticks + 1):
        step = steps_train[0] + i * (steps_train[-1] - steps_train[0]) / ticks
        x, y = transform(step, min_ppl)
        cv2.line(img, (x, height - margin - 5), (x, height - margin + 5), (0, 0, 0), 1)
        cv2.putText(
            img,
            f"{int(step)}",
            (x - 10, height - margin + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
        )

    # Y-axis ticks (perplexity)
    for i in range(ticks + 1):
        ppl = min_ppl + i * (max_ppl - min_ppl) / ticks
        x, y = transform(steps_train[0], ppl)
        cv2.line(img, (margin - 5, y), (margin + 5, y), (0, 0, 0), 1)
        cv2.putText(
            img,
            f"{ppl:.1f}",
            (5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
        )

    # Train perplexity (blue)
    for i in range(1, len(train_ppl)):
        cv2.line(
            img,
            transform(steps_train[i - 1], train_ppl[i - 1]),
            transform(steps_train[i], train_ppl[i]),
            (255, 0, 0),
            2,
        )

    # Validation perplexity (orange)
    for i in range(1, len(val_ppl)):
        cv2.line(
            img,
            transform(steps_val[i - 1], val_ppl[i - 1]),
            transform(steps_val[i], val_ppl[i]),
            (0, 165, 255),
            2,
        )

    # Labels
    cv2.putText(img, "Train Perplexity (Blue)", (margin, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(img, "Validation Perplexity (Orange)", (margin + 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.putText(img, "Steps", (width // 2 - 30, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Perplexity", (10, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imwrite(save_path, img)
    print(f"Perplexity graph saved to {save_path}")


