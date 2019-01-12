import inspect
import os

def get_relative_path(file_path):
		dirname = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
		file_path = os.path.join(dirname, file_path)

		return file_path
