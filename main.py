from src.pipeline import run_pipeline
from src.plots import plot_all
from src.model import train_model

def main():
    df = run_pipeline()

    plot_all(df)
    train_model(df)

if __name__ == "__main__":
    main()