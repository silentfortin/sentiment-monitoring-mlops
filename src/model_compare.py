import json
import os
import shutil

def compare_and_promote(new_model_dir="models/roberta_sentiment_model",
                        prod_model_dir="models/production_model",
                        base_metrics_path="models/base_model_metrics.json"):

    new_metrics = json.load(open(os.path.join(new_model_dir, "metrics.json")))
    new_acc = new_metrics["accuracy"]

    if not os.path.exists(prod_model_dir):
        print("No production model found — using base metrics.")
        if not os.path.exists(base_metrics_path):
            print("Base metrics not found — promoting current model.")
            shutil.copytree(new_model_dir, prod_model_dir, dirs_exist_ok=True)
            return
        old_acc = json.load(open(base_metrics_path))["accuracy"]
    else:
        old_acc = json.load(open(os.path.join(prod_model_dir, "metrics.json")))["accuracy"]

    print(f"Old accuracy: {old_acc:.3f} | New accuracy: {new_acc:.3f}")

    if new_acc > old_acc:
        print(f"Promoting new model ({new_acc:.3f} > {old_acc:.3f})")
        shutil.copytree(new_model_dir, prod_model_dir, dirs_exist_ok=True)
    else:
        print(f"Keeping old model ({new_acc:.3f} >= {old_acc:.3f})")
