import andes
import pandas as pd

path = andes.get_case('ieee39/ieee39.xlsx')

# Load all sheets at once as a dict of DataFrames
all_sheets = pd.read_excel(path, sheet_name=None)

print("Workbook Sheets and Their Column Headers:")
for sheet_name, df in all_sheets.items():
    print(f"Sheet: {sheet_name}")
    print(f"Columns: {list(df.columns)}")
    print("\n")


ssa = andes.load(
    path,
    setup=True,
    no_output=True,
)
ssa.PFlow.run()

# 1. Load the workbook and list sheets
xls = pd.ExcelFile(path)
first_sheet = xls.sheet_names[0]

# 2. Parse the first sheet to get its column headers
df0 = xls.parse(first_sheet)
headers = df0.columns.tolist()

# 3. Make sure there are at least 7 columns
if len(headers) < 7:
    raise ValueError(f"Sheet '{first_sheet}' Only Has {len(headers)} Columns -- Can't Get the 7th Header")

# 4. Get the 7th header (index 6)
seventh_header = headers[6]

# 5. Pull the same field from the loaded ANDES model
model = getattr(ssa, first_sheet, None)
if model is None or not hasattr(model, seventh_header):
    raise AttributeError(
        f"Model '{first_sheet}' does not expose field '{seventh_header}' in the loaded ANDES case."
    )

values = getattr(model, seventh_header).v

print(f"1st Sheet: '{first_sheet}'")
print(f"7th Header: '{seventh_header}'")
print(f"Values ({first_sheet}.{seventh_header}.v):\n", values)


# Load the IEEE 39-bus workbook, list all sheets with their column headers, run a power-flow simulation, then retrieve and display the values of the 7th column from the first sheet using ANDES.
