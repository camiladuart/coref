#Coref loader:
Repository created to load coreference data (ontoNotes).

## Converter OntoNotes → JSONLines
```powershell
python .\coref\ontonotes_to_jsonlines.py `
  --input_dir "C:\Users\PC\ontonotes-release-5.0" `
  --output_dir "C:\Users\PC\coref_data\ontonotes"
