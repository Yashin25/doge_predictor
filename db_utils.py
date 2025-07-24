import gspread
from google.oauth2.service_account import Credentials

def save_predictions_to_gsheet(df, sheet_name="Predictions", worksheet_name="Sheet1", creds_path="service_account.json"):
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open(sheet_name)
        worksheet = sh.worksheet(worksheet_name)
        rows = df.reset_index().values.tolist()
        worksheet.append_rows(rows, value_input_option="USER_ENTERED")
    except Exception as e:
        print(f"Błąd zapisu do Google Sheets: {e}")