import urllib, urllib.request
import xml.etree.ElementTree as ET
import json
from datetime import datetime


url = 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1'
response_data = urllib.request.urlopen(url)
xml_data = response_data.read().decode('utf-8')

if response_data.getcode() == 200:
    print('Success')
else:
    print('Failed')

# Parse the XML data
namespace = {"arxiv": "http://arxiv.org/schemas/atom", "default": "http://www.w3.org/2005/Atom"}
root = ET.fromstring(xml_data)
entry = root.find("default:entry", namespace)

# Convert dates to required formats
updated_date = entry.find("default:updated", namespace).text
create_date = datetime.strptime(updated_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%a, %d %b %Y %H:%M:%S GMT")

# Build the JSON result
result = {
    "id": entry.find("default:id", namespace).text.split("/")[-1],
    "submitter": entry.find("default:author/default:name", namespace).text,
    "authors": ", ".join(author.find("default:name", namespace).text for author in entry.findall("default:author", namespace)),
    "title": entry.find("default:title", namespace).text.strip(),
    "comments": entry.find("arxiv:comment", namespace).text if entry.find("arxiv:comment", namespace) is not None else None,
    "journal-ref": entry.find("arxiv:journal_ref", namespace).text if entry.find("arxiv:journal_ref", namespace) is not None else None,
    "doi": entry.find("arxiv:doi", namespace).text if entry.find("arxiv:doi", namespace) is not None else None,
    "report-no": None,
    "categories": entry.find("arxiv:primary_category", namespace).attrib["term"],
    "license": None,
    "abstract": entry.find("default:summary", namespace).text.strip(),
    "versions": [
        {
            "version": "v1",
            "create": create_date,
        }
    ],
    "update_date": datetime.fromisoformat(root.find("default:updated", namespace).text).strftime("%Y-%m-%d"),
    "authors_parsed": [
        author.find("default:name", namespace).text
        for author in entry.findall("default:author", namespace)
    ]
}

# Convert to JSON
json_result = json.dumps(result)

# Append JSON result to file
output_file = 'arxiv_papers.json'
with open(output_file, 'a') as f:
    f.write(json_result + '\n')

print('Data appended to', output_file)