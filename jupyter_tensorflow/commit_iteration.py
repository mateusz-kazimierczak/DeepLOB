import argparse
from pathlib import Path

from dulwich import porcelain
from dulwich.repo import Repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--message", required=True)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    repo = Repo(str(repo_path))

    for rel_path in args.paths:
        porcelain.add(repo.path, paths=[rel_path.encode("utf-8")])

    commit_id = porcelain.commit(
        repo.path,
        message=args.message.encode("utf-8"),
    )
    print(commit_id.decode("ascii"))


if __name__ == "__main__":
    main()
