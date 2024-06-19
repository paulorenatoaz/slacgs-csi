import os


def is_running_on_colab():
	"""Check if the code is running on Google Colab."""
	try:
		import google.colab
		return True
	except ImportError:
		return False


if is_running_on_colab():
	PROJECT_DIR = '/content/slacgs-csi'
else:
	PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
