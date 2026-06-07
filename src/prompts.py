def get_clean_corrupted_pair():
    clean = "The Eiffel Tower is located in the city of Paris"
    corrupted = "The Eiffel Tower is located in the city of London"
    return clean, corrupted

if __name__ == "__main__":
    clean, corrupted = get_clean_corrupted_pair()
    print(f"clean: {clean!r}")
    print(f"corrupted: {corrupted!r}")