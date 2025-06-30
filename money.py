import pandas as pd
from openai import OpenAI

# =============== CONFIGURATION ===============

CSV_BLACKROCK = "test_etf.csv"
CSV_THRIVENT = "test_ga.csv"

MODEL_NAME = "gpt-4o"
MAX_ROWS = 50

# =============== FUNCTIONS ===============

def clean_csv_generic(raw_path):
    """
    Cleans a messy CSV with all data in a single column.
    Adjust if your  file has different formatting.
    """
    print(f"Reading raw file: {raw_path}")
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Total lines read: {len(lines)}")

    # Find the header row
    header_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("As Of Date"):
            header_index = i
            break

    if header_index is None:
        raise ValueError("Could not find the header row starting with 'As Of Date'.")

    headers = [h.strip() for h in lines[header_index].strip().split(",")]
    print(f"Parsed headers for {raw_path}: {headers}")

    data_rows = []
    for line in lines[header_index + 1:]:
        if not line.strip():
            continue
        row = [cell.strip() for cell in line.strip().split(",")]
        if len(row) < len(headers):
            row.extend([""] * (len(headers) - len(row)))
        elif len(row) > len(headers):
            row = row[:len(headers)]
        data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=headers)
    df.dropna(how="all", inplace=True)
    return df

def df_to_text(df, institution_name, max_rows=MAX_ROWS):
    """
    Converts DataFrame to readable text.
    """
    if len(df) > max_rows:
        df = df.head(max_rows)
        note = f"(Showing only first {max_rows} rows.)\n\n"
    else:
        note = ""
    return f"Data from {institution_name}:\n{note}" + df.to_string(index=False)

# =============== MAIN SCRIPT ===============

def main():
    print("Cleaning test_etf data...")
    df_blackrock = clean_csv_generic(CSV_BLACKROCK)

    print("Cleaning test_ga data...")
    df_thrivent = clean_csv_generic(CSV_THRIVENT)

    print("Converting cleaned data to text...")
    text_etf = df_to_text(df_etf, "Test ETF Model Portfolio")
    text_ga = df_to_text(df_ga, "Test GA Model Portfolio")

    # Prompt for the model
    prompt = f"""
You are an expert financial analyst.

Below are holdings data from two funds.

{text_etf}


{text_ga}

Please answer the following questions:
Only compare the 60/40 model portfolios from each institution.
1. What are the top 3 holdings where test ETF and test GA differ most in allocation percentage?
2. Which asset classes are most heavily weighted by each institution?
3. What does the turnover since the last rebalance suggest about each portfolio's strategy?
4. Summarize in plain language the main differences between these two portfolios.
5. If an investor prefers stability over aggressive rebalancing, which model may be more suitable and why?
6. what other things can you say about these portfolios based on the data provided that i haven't asked about?
Use only the data provided. Do not speculate beyond this data.


after you asnwer all of that provide some insite where you actually use outside data to provide more context to the analysis.
"""

    print("Initializing OpenAI client...")
    client = OpenAI()

    print("Requesting comparison analysis from the model...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful financial data analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    print("\n==========================")
    print("📊 COMPARISON ANALYSIS 📊")
    print("==========================\n")
    print(answer)

# =============== ENTRY POINT ===============

if __name__ == "__main__":
    main()
