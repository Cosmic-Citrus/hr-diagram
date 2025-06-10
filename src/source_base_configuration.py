

class BaseSourceConfiguration():

	def __init__(self):
		super().__init__()
		self._name = None
		self._title = None
		self._author = None
		self._general_url = None
		self._personal_url = None
		self._data_url = None
		self._license = None
		self._original_source = None
		self._path_to_file = None
		self._is_web_scraped = None
		self._df = None

	@property
	def name(self):
		return self._name

	@property
	def title(self):
		return self._title
	
	@property
	def author(self):
		return self._author
	
	@property
	def general_url(self):
		return self._general_url
	
	@property
	def personal_url(self):
		return self._personal_url
	
	@property
	def data_url(self):
		return self._data_url
	
	@property
	def license(self):
		return self._license

	@property
	def original_source(self):
		return self._original_source
	
	@property
	def path_to_file(self):
		return self._path_to_file
	
	@property
	def is_web_scraped(self):
		return self._is_web_scraped
	
	@property
	def df(self):
		return self._df
	
	def initialize_source_information(self, name, title, author, general_url, personal_url, data_url, license, original_source):
		self._name = name
		self._title = title
		self._author = author
		self._general_url = general_url
		self._personal_url = personal_url
		self._data_url = data_url
		self._license = license
		self._original_source = original_source

	def save_df_as_csv(self, path_to_directory):
		if not isinstance(path_to_directory, str):
			raise ValueError("invalid type(path_to_directory): {}".format(type(path_to_directory)))
		path_to_csv = "{}/{}.csv".format(
			path_to_directory,
			self.name.replace(
				" ",
				"_"))
		self.df.to_csv(
			path_to_csv,
			index=False)

	@staticmethod
	def initialize_df(*args, **kwargs):
		raise ValueError("this method should be over-written by a child class")

class SourceConfiguration(BaseSourceConfiguration):

	def __init__(self, name, title, author, general_url, personal_url, data_url, license, original_source):
		super().__init__()
		self.initialize_source_information(
			name=name,
			title=title,
			author=author,
			general_url=general_url,
			personal_url=personal_url,
			data_url=data_url,
			license=license,
			original_source=original_source)

	def initialize(self, path_to_file=None, path_to_directory=None):
		self.initialize_df(
			path_to_file=path_to_file)
		if path_to_directory is not None:
			self.save_df_as_csv(
				path_to_directory=path_to_directory)
			
##