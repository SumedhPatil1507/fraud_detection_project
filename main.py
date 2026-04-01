from src.pipeline import run_pipeline
from src.model import train_model

def main():
    df = run_pipeline()
    train_model(df)

if __name__ == "__main__":
    main()