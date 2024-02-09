import os
import json
import pytextable

# If 'repo' is None, then product is considered proprietary
methodInfo = {
    "aim": {"name":"AIM",
            "repo":"https://github.com/opendp/smartnoise-sdk",
            "cite":"mckenna2022aim",
            "web":"https://opendp.org/",
            },
    "aindo-synth": {"name":"Aindo",
            "repo":None,
            "cite":None,
            "web":"https://www.aindo.com/",
            },
    "Anonos Data Embassy SDK": {"name":"Anonos",
            "repo":None,
            "cite":None,
            "web":"https://www.anonos.com/",
            },
    "cart": {"name":"CART",
            "repo":"https://CRAN.R-project.org/package=synthpop",
            "cite":"nowok2016synthpop",
            "web":"https://synthpop.org.uk/",
            },
    "DPHist": {"name":"Tumult",
            "repo":None,
            "cite":None,
            "web":"https://www.tmlt.io/",
            },
    "Genetic SD": {"name":"Genetic",
            "repo":"https://github.com/giusevtr/private_gsd",
            "cite":"liu2023generating",
            "web":None,
            },
    "kanonymity": {"name":"K6-Anon",
            "repo":"https://github.com/sdcTools/sdcMicro",
            "cite":"templ2015statistical",
            "web":None,
            },
    "MostlyAI SD": {"name":"MostlyAI",
            "repo":None,
            "cite":None,
            "web":"https://mostly.ai/",
            },
    "MWEM+PGM": {"name":"mwem+pgm",
            "repo":None,
            "cite":"mckenna2019graphical",
            "web":None,
            },
    "pategan": {"name":"Pategan",
            "repo":"https://github.com/PerceptionLab-DurhamUniversity/pategan",
            "cite":"jordon2018pate",
            "web":None,
            },
    "Sarus SDG": {"name":"Sarus",
            "repo":None,
            "cite":"canale2022generative",
            "web":"https://www.sarus.tech/",
            },
    "ctgan": {"name":"CTGAN",
            "repo":"https://github.com/sdv-dev/SDV",
            "cite":"xu2019modeling",
            "web":"https://sdv.dev/",
            },
    "smote": {"name":"SMOTE",
            "repo":"https://github.com/ut-dallas-dspl-lab/AI-Fairness",
            "cite":"zhou2023improving",
            "web":None,
            },
    "subsample_40pcnt": {"name":"Sample40",
            "repo":None,
            "cite":"acsBasics2021",
            "web":None,
            },
    "SynDiffix": {"name":"SynDiffix",
            "repo":"https://github.com/diffix/syndiffix",
            "cite":"francis2023syndiffix",
            "web":"https://www.open-diffix.org/",
            },
    "YData Fabric Synthesizers": {"name":"YData",
            "repo":"https://github.com/ydataai/ydata-synthetic",
            "cite":None,
            "web":"https://ydata.ai/",
            },
}

keyOrder = [
    "AIM",
    "Aindo",
    "Anonos",
    "CART",
    "Tumult",
    "Genetic",
    "K6-Anon",
    "MostlyAI",
    "MWEM+PGM",
    "PATEGAN",
    "CTGAN",
    "SMOTE",
    "Sample40",
    "SynDiffix",
    "YData",
]

class ReportJsonReader:
    def __init__(self, algData):
        self.algName = algData['data_description']['deid']['labels']['algorithm name']
        self.team = algData['data_description']['deid']['labels']['team']
        self.info = methodInfo[self.algName]
        self.name = self.info['name']
        self.repo = self.info['repo']
        self.cite = self.info['cite']
        self.web = self.info['web']
        self.features = algData['data_description']['deid']['labels']['features list'].split(',')
        self.ncol = len(self.features)
        self.dp = True if algData['data_description']['deid']['labels']['privacy category'] == 'dp' else False
        self.epsilon = algData['data_description']['deid']['labels']['epsilon'] if self.dp is True else None

class GatherResults:
    def __init__(self, resultsDir, outDir):
        self.resultsDir = resultsDir
        self.outDir = outDir
        self.algsBase = {}
        self.algDirs = [d for d in os.listdir(self.resultsDir) if os.path.isdir(os.path.join(self.resultsDir, d))]
        self.res = {}
        for dir in self.algDirs:
            resPath = os.path.join(self.resultsDir, dir, 'report.json')
            with open(resPath, 'r') as f:
                algData = json.load(f)
            rjr = ReportJsonReader(algData)
            self.res[rjr.name] = rjr

    def makeInfoTable(self):
        header = ["Algorithm", "Columns", "Epsilon", "Cite", "More Info"]
        body = []
        for alg in keyOrder:
            rjr = self.res[alg]
            row = []
            row.append(rjr.name)
            row.append(rjr.ncol)
            row.append(rjr.epsilon if rjr.epsilon is not None else '---')
            row.append("\\cite{" + rjr.cite +"}" if rjr.cite is not None else '---')
            if rjr.repo is None and rjr.web is None:
                row.append('')
                body.append(row)
            elif rjr.repo is not None:
                row.append("Code: \\small{\\url{" + rjr.repo + "}}")
                body.append(row)
                if rjr.web is not None:
                    row = ['','','','', "URL: \\small{\\url{" + rjr.web + "}}"]
                    body.append(row)
            elif rjr.web is not None:
                row.append("URL: \\small{\\url{" + rjr.web + "}}")
                body.append(row)
        outPath = os.path.join(self.outDir, 'infoTable.tex')
        with open(outPath, 'w') as f:
            f.write(pytextable.tostring(body, header=header))

if __name__ == "__main__":
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    gr = GatherResults('results', 'outputs')
    gr.makeInfoTable()
