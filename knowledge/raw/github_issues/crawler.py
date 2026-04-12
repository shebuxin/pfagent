import os
import csv
import requests
from typing import List, Dict
from time import sleep

class GitHubIssueCrawler:
    API_ROOT = "https://api.github.com"

    def __init__(self, owner, repo, token = None):
        """
        Args:
            owner: GitHub user or org name
            repo: Repository name
            token: Optional personal access token (for higher rate limits)
        """
        self.owner = owner
        self.repo = repo
        self.token = token or os.getenv("GITHUB_TOKEN")

    def _headers(self):
        hdr = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            hdr["Authorization"] = f"token {self.token}"
        return hdr

    def _paginate(self, url, params = None):
        params = params or {}
        params.setdefault("per_page", 100)
        while url:
            resp = requests.get(url, headers=self._headers(), params=params)
            resp.raise_for_status()
            data = resp.json()
            for item in data:
                yield item

            # find next page
            links = requests.utils.parse_header_links(resp.headers.get("Link", ""))
            next_link = next((l["url"] for l in links if l.get("rel") == "next"), None)
            url = next_link
            params = None  
            if not self.token:
                sleep(1)

    def fetch_issues(self, state = "all", exclude_prs = True):
        url = f"{self.API_ROOT}/repos/{self.owner}/{self.repo}/issues"
        raw = list(self._paginate(url, params={"state": state}))
        if exclude_prs:
            return [i for i in raw if "pull_request" not in i]
        else:
            return raw

    def export_to_csv(self, issues, filename):
        print(f"Exporting to CSV: {filename}")

        fieldnames = [
            'number', 'title', 'state', 'author', 'created_at',
            'updated_at', 'comments', 'labels', 'assignees', 'url', 'body'
        ]

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for issue in issues:
                writer.writerow({
                    'number': issue['number'],
                    'title': issue['title'],
                    'state': issue['state'],
                    'author': issue['user']['login'],
                    'created_at': issue['created_at'],
                    'updated_at': issue['updated_at'],
                    'comments': issue['comments'],
                    'labels': ', '.join(lbl['name'] for lbl in issue.get('labels', [])),
                    'assignees': ', '.join(a['login'] for a in issue.get('assignees', [])),
                    'url': issue['html_url'],
                    'body': (issue.get('body') or '')
                             .replace('\n', ' ')
                             .replace('\r', '')[:500]
                })

if __name__ == "__main__":
    OWNER = "CURENT"
    REPO  = "andes"
    TOKEN = os.getenv("GITHUB_TOKEN")

    crawler = GitHubIssueCrawler(OWNER, REPO, token=TOKEN)
    issues = crawler.fetch_issues(state="all", exclude_prs = True)
    crawler.export_to_csv(issues, filename="andes_github_issues.csv")
    print(f"Done - {len(issues)} Issues Written.")    
