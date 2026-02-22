import andes
import pandas as pd


path = "ieee39_base.xlsx"

# Load all sheets at once as a dict of DataFrames
all_sheets = pd.read_excel(path, sheet_name=None)

print("Workbook Sheets and Their Column Headers:")
for sheet_name, df in all_sheets.items():
    print(f"Sheet: {sheet_name}")
    print(f"Columns: {list(df.columns)}")
    print("\n")


ssa = andes.load(
    path,
    setup=False,     # If setup=False, need to call ss.setup() before running the simulation
    no_output=True, 
    default_config=False
)

ssa.setup()
ssa.PFlow.run()


# 1. Load the workbook and list sheets
xls = pd.ExcelFile("ieee39_base.xlsx")
sheet_names = xls.sheet_names
first_sheet = sheet_names[0]

# 2. Parse the first sheet to get its column headers
df0 = xls.parse(first_sheet)
headers = df0.columns.tolist()

# 3. Make sure there are at least 7 columns
if len(headers) < 7:
    raise ValueError(f"Sheet '{first_sheet}' Only Has {len(headers)} Columns -- Can't Get the 7th Header")

# 4. Get the 7th header (index 6)
seventh_header = headers[6]

# 5. Finally, pull its `.v` array from ANDES
values = getattr(getattr(ssa, first_sheet), seventh_header).v

print(f"1st Sheet: '{first_sheet}'")
print(f"7th Header: '{seventh_header}'")
print(f"Values ({first_sheet}.{seventh_header}.v):\n", values)


# Load the IEEE 39-bus workbook, list all sheets with their column headers, run a power-flow simulation, then retrieve and display the values of the 7th column from the first sheet using ANDES.