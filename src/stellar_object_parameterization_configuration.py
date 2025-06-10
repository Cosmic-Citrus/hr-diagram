from data_processing_configuration import DataProcessingConfiguration
from plotter_hr_diagram_configuration import HRDiagramViewer


class BaseStellarObjectParameterizationConfiguration(DataProcessingConfiguration):

	def __init__(self):
		super().__init__()

	def get_initialized_plotter(self):
		self.verify_visual_settings()
		plotter = HRDiagramViewer()
		plotter.initialize_visual_settings(
			tick_size=self.visual_settings.tick_size,
			label_size=self.visual_settings.label_size,
			text_size=self.visual_settings.text_size,
			cell_size=self.visual_settings.cell_size,
			title_size=self.visual_settings.title_size)
		plotter.update_save_directory(
			path_to_save_directory=self.visual_settings.path_to_save_directory)
		return plotter

class StellarObjectParameterizationConfiguration(BaseStellarObjectParameterizationConfiguration):

	def __init__(self):
		super().__init__()

	def initialize(self, path_to_hyg_data_file=None, path_to_classification_file=None, path_to_data_directory=None):
		self.initialize_visual_settings()
		self.initialize_classification_parameterization()
		self.initialize_sources(
			path_to_hyg_data_file=path_to_hyg_data_file,
			path_to_classification_file=path_to_classification_file,
			path_to_data_directory=path_to_data_directory)
		self.initialize_sun()
		self.initialize_data()

	def view_blank_hr_diagram(self, *args, **kwargs):
		plotter = self.get_initialized_plotter()
		plotter.view_blank_hr_diagram(
			self,
			*args,
			**kwargs)

	def view_hr_diagram(self, *args, **kwargs):
		plotter = self.get_initialized_plotter()
		plotter.view_hr_diagram(
			self,
			*args,
			**kwargs)

	def view_two_dimensional_histogram(self, x_bin_selection_method, y_bin_selection_method, *args, **kwargs):
		plotter = self.get_initialized_plotter()
		plotter.view_two_dimensional_histogram(
			self,
			x_bin_selection_method,
			y_bin_selection_method,
			*args,
			**kwargs)

##