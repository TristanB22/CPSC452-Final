# standard library imports
import os
import random
import shutil
import time
from typing import List, Union

# third party imports
import applescript
import requests
from bs4 import BeautifulSoup

# local imports


class UIUCAirfoilDownloader:
    def __init__(
        self,
        use_website: bool = False,
        save_folder: str = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "raw"
        ),
    ):
        self.use_website = use_website
        self.save_folder = save_folder

    def scrape_website(self) -> Union[List[str], None]:
        script = """
            tell application "Google Chrome"
                set currentTab to active tab of front window
                execute currentTab javascript "document.documentElement.outerHTML"
            end tell
        """

        try:
            result = applescript.run(script)
            if result:
                if result.out:
                    soup = BeautifulSoup(result.out, "lxml")
                    a_tags = soup.find_all("a", href=True)
                    dat_links = [
                        a["href"] for a in a_tags if a["href"].endswith(".dat")
                    ]

                    return dat_links
                else:
                    raise ValueError(f"Failed to capture HTML: {result.err}")
        except Exception as e:
            raise e

    def get_file_links(self) -> List[str]:
        if self.use_website:
            dat_links = self.scrape_website()

            if not dat_links:
                raise ValueError("No .dat files found on the website.")

            dat_links = [
                "https://m-selig.ae.illinois.edu/ads/" + d_link for d_link in dat_links
            ]
        else:
            dat_links = open(
                os.path.join(os.path.dirname(__file__), "uiuc_data_links.txt"), "r"
            ).readlines()
            dat_links = [link.strip() for link in dat_links]

        return dat_links

    def download_files(self, urls: List[str]) -> None:
        os.makedirs(self.save_folder, exist_ok=True)

        for url in urls:
            time.sleep(0.5)

            # randomly wait
            if random.random() < 0.1:
                time.sleep(1)

            filename = url.split("/")[-1]
            save_path = os.path.join(self.save_folder, filename)

            # get the data
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                # write the output
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {filename} to {save_path}")
            else:
                print(f"Failed to download {url}")

        print("Download process completed.")

    def __call__(self) -> None:
        dat_links = self.get_file_links()
        self.download_files(dat_links)

        # zip the folder for easy transport + GitHub upload
        shutil.make_archive(
            self.save_folder,
            "zip",
            self.save_folder,
        )


if __name__ == "__main__":
    downloader = UIUCAirfoilDownloader(use_website=False)
    downloader()
