from experiment import compare_algorithms_from_file

if __name__ == "__main__":
    files = [
         ("knapsack_data/data/ks_400_0", 2000),
    ]

    for file_path, episodes in files:
        print(f"\n=== DOSYA: {file_path} ===")
        compare_algorithms_from_file(file_path=file_path, episodes=episodes)