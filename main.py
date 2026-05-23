from experiment import compare_algorithms_from_file

if __name__ == "__main__":
    files = [
        ("knapsack_data/data/ks_50_0", 1200),
        ("knapsack_data/data/ks_100_0", 1800),
        ("knapsack_data/data/ks_200_0", 2500),
        ("knapsack_data/data/ks_400_0", 2000),
        ("knapsack_data/data/ks_1000_0", 2000),
    ]

    for file_path, episodes in files:
        print(f"\n=== DOSYA: {file_path} ===")
        compare_algorithms_from_file(file_path=file_path, episodes=episodes)