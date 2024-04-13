__VERSION__ = "0.0.3"

def clean_repo(repo_name: str) -> None:
    """Clean the repository"""
    import os

    if os.name == "nt":
        command = f"rmdir build /s /q & rmdir /s /q {repo_name}.egg-info & rmdir /s /q dist"
    elif os.name == "posix":
        command = f"rm -r build & rm -r  {repo_name}.egg-info & rm -r dist"
    else:
        command = ""

    os.system(command)

def install_package(package: str):
    """Install desired package"""
    import pip

    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])