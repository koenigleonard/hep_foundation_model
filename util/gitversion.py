import subprocess
print(subprocess.check_output(["git", "rev-parse", "HEAD"]).decode())