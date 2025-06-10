from source_base_configuration import SourceConfiguration
import pandas as pd
import requests
from bs4 import BeautifulSoup

class DataConfigurationStellarParametersTable(SourceConfiguration):

	def __init__(self):
		super().__init__(
			name="Stellar Classification Table - sorted by HR Class",
			title="Stellar Classification Table",
			author="Landon Curt Noll",
			general_url="http://www.isthe.com/chongo/tech/astro/HR-temp-mass-table-byhrclass.html",
			personal_url="http://www.isthe.com/chongo/index.html",
			data_url=None, ## parse web-scraper
			license="Public Domain (no usage restrictions, no warranty)",
			original_source="SpaceGear.org")

	def initialize_df(self, path_to_file):
		if path_to_file is None:
			is_web_scraped = True
			target_path = self.general_url[:]
			response = requests.get(
				target_path)
			response.raise_for_status()
			soup = BeautifulSoup(
				response.content,
				"html.parser")
			table = soup.find(
				"table",
				{"border" : "1"})
			headers = list()
			header_row = table.find(
				"tr")
			for header in header_row.find_all("th"):
				headers.append(
					header.text.strip())
			rows = list()
			for row in table.find_all("tr")[1:]: ## skip row of headers
				row_data = [
					data.text.strip()
						for data in row.find_all(
							"td")]
				if row_data:
					rows.append(
						row_data)
			df = pd.DataFrame(
				rows,
				columns=headers)
			df = df[~df.apply(
				lambda row: row.astype(str).str.strip().eq("").all(),
				axis=1)]
			new_header = df.iloc[0]
			df = df[1:]
			# df.columns = new_header
			# df.columns = df.columns.str.strip()
			df[["R", "G", "B"]] = df.iloc[:, -1].str.split(
				" ",
				expand=True)
			index_at_combined_rgb_column = int(
				-1 * (len("rgb") + 1))
			df = df.drop(
				columns=df.columns[index_at_combined_rgb_column])
			df = df.reset_index(
				drop=True)
		elif isinstance(path_to_file, str):
			is_web_scraped = False
			target_path = path_to_file[:]
			df = pd.read_csv(
				target_path)
		else:
			raise ValueError("invalid type(path_to_file): {}".format(type(path_to_file)))
		df.dropna()
		self._is_web_scraped = is_web_scraped
		self._df = df

##