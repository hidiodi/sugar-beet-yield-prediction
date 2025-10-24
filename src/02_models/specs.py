import psutil

# Get the number of physical cores (most reliable for setting n_workers)
physical_cores = psutil.cpu_count(logical=False)
# Get the number of logical cores (includes hyper-threading)
logical_cores = psutil.cpu_count(logical=True)

print(f"This machine has {physical_cores} physical CPU cores.")
print(f"This machine has {logical_cores} logical CPU cores (with hyper-threading).")

# Recommendation:
print(f"\nRecommended setting: n_workers = {physical_cores}")