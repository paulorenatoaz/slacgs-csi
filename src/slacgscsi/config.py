import os
import sys


def is_running_on_colab():
	"""Check if the code is running on Google Colab."""
	try:
		import google.colab
		return True
	except ImportError:
		return False


# Add the 'src' directory to the sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
print("Source directory added to sys.path:", src_path)
sys.path.append(src_path)

if is_running_on_colab():
	PROJECT_DIR = '/content/slacgs-csi'
else:
	PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
