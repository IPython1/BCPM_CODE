import json
import os
import sys


def generate_dict_from_file(path, file_name):
    user_interaction = {}
    with open(os.path.join(path, file_name), encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue
            parts = row.split()
            user, item = int(parts[0]), int(parts[1])
            if user not in user_interaction:
                user_interaction[user] = [item]
            elif item not in user_interaction[user]:
                user_interaction[user].append(item)
    return user_interaction


def save_dict(data, path, file_name):
    str_dict = {str(key): value for key, value in data.items()}
    with open(os.path.join(path, file_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(str_dict))


def generate_interactions(path):
    test_dict = generate_dict_from_file(path, "test.txt")
    validation_dict = generate_dict_from_file(path, "validation.txt")

    held_out_pairs = set()
    for user, items in test_dict.items():
        for item in items:
            held_out_pairs.add((user, item))
    for user, items in validation_dict.items():
        for item in items:
            held_out_pairs.add((user, item))

    browse_dict = generate_dict_from_file(path, "browse.txt")
    save_dict(browse_dict, path, "browse_dict.txt")

    favorite_dict = generate_dict_from_file(path, "favorite.txt")
    save_dict(favorite_dict, path, "favorite_dict.txt")

    study_train_dict = {}
    with open(os.path.join(path, "study.txt"), encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue
            parts = row.split()
            user, item = int(parts[0]), int(parts[1])
            if (user, item) in held_out_pairs:
                continue
            if user not in study_train_dict:
                study_train_dict[user] = [item]
            elif item not in study_train_dict[user]:
                study_train_dict[user].append(item)

    save_dict(study_train_dict, path, "study_dict.txt")

    with open(os.path.join(path, "study.txt"), "w", encoding="utf-8") as f:
        for user, items in study_train_dict.items():
            for item in items:
                f.write(f"{user}\t{item}\n")

    save_dict(validation_dict, path, "validation_dict.txt")
    save_dict(test_dict, path, "test_dict.txt")

    return browse_dict, favorite_dict, study_train_dict, validation_dict, test_dict


def generate_all_interactions(path, browse_dict, favorite_dict, study_dict):
    all_dict = {}
    for behavior_dict in [browse_dict, favorite_dict, study_dict]:
        for user, items in behavior_dict.items():
            user_key = str(user)
            if user_key not in all_dict:
                all_dict[user_key] = list(items)
            else:
                total_items = all_dict[user_key]
                total_items.extend(items)
                all_dict[user_key] = sorted(list(set(total_items)))

    with open(os.path.join(path, "all.txt"), "w", encoding="utf-8") as all_file, open(
        os.path.join(path, "all_dict.txt"), "w", encoding="utf-8"
    ) as all_dict_file:
        for user, items in all_dict.items():
            for item in items:
                all_file.write(f"{int(user)} {item}\n")
        all_dict_file.write(json.dumps(all_dict))


def generate_count(path, browse_dict, favorite_dict, study_dict, validation_dict, test_dict):
    max_user = 0
    max_item = 0
    for behavior_dict in [
        browse_dict,
        favorite_dict,
        study_dict,
        validation_dict,
        test_dict,
    ]:
        for user, items in behavior_dict.items():
            user = int(user)
            if user > max_user:
                max_user = user
            for item in items:
                if item > max_item:
                    max_item = item

    count = {"user": max_user, "item": max_item}
    with open(os.path.join(path, "count.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(count))
    print(f"count.txt -> user: {max_user}, item: {max_item}")


def generate_positive_sampling(path):
    behaviors = ["browse", "favorite", "study"]
    with open(os.path.join(path, "pos_sampling.txt"), "w", encoding="utf-8") as f:
        for behavior_index, behavior in enumerate(behaviors):
            with open(
                os.path.join(path, behavior + "_dict.txt"), encoding="utf-8"
            ) as behavior_file:
                behavior_dict = json.load(behavior_file)
                for user, items in behavior_dict.items():
                    for item in items:
                        f.write(f"{user} {item} {behavior_index} 1\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    subdir = sys.argv[1] if len(sys.argv) > 1 else "MOOCCube"
    current_path = os.path.join(base_dir, subdir)

    print("Step 1: building behavior interaction dictionaries ...")
    browse_dict, favorite_dict, study_dict, validation_dict, test_dict = (
        generate_interactions(current_path)
    )

    print("Step 2: building all.txt / all_dict.txt ...")
    generate_all_interactions(current_path, browse_dict, favorite_dict, study_dict)

    print("Step 3: building count.txt ...")
    generate_count(
        current_path,
        browse_dict,
        favorite_dict,
        study_dict,
        validation_dict,
        test_dict,
    )

    print("Step 4: building pos_sampling.txt ...")
    generate_positive_sampling(current_path)

    print("Done.")
