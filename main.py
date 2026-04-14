from src.data_loader import load_alzheimers, load_parkinsons_v2 # load_autism

def main():
    print("Testing data loaders after Sakar 2018 dataset pivot...")
    
    datasets = {
        "OASIS Alzheimer's": load_alzheimers,
        "Parkinson's (Sakar)": load_parkinsons_v2,
       # "Autism Screening": load_autism
    }

    for name, loader_func in datasets.items():
        try:
            X, y = loader_func()
            print(f"[OK] {name} loaded successfully. Shape: {X.shape}")
        except Exception as e:
            print(f"[ERROR] Failed on {name}: {e}")

if __name__ == "__main__":
    main()