/** @OnlyCurrentDoc */

function AddRelColandHelpNote() {
  var spreadsheet = SpreadsheetApp.getActive();
  spreadsheet.getRange('D1').activate();
  spreadsheet.getCurrentCell().setValue('relevance_level');
  spreadsheet.getActiveSheet().setColumnWidth(4, 115);
  spreadsheet.getRange('C1').setNote("When you identify an 'utterance_text' as a check-worthy claim, set the value of this field to TRUE.");
  spreadsheet.getRange('E9').activate();
  spreadsheet.getRange('D1').setNote("When 'is_check_worthy_claim' is set to TRUE, this field prompts you to select a relevance level. It should remain blank when 'is_check_worthy_claim' is FALSE.");
};


function ProtectFieldsInDuplicates() {
  var spreadsheet = SpreadsheetApp.getActive();
  var protection = spreadsheet.getRange('1:1').protect().setDescription('Protect Header Row');
  protection.removeEditors(protection.getEditors());

  protection = spreadsheet.getRange('A:B').protect().setDescription('Protect Header columns');
  protection.removeEditors(protection.getEditors());
};


function MakeDuplicateSheets() {
  var spreadsheet = SpreadsheetApp.getActive();
  var sheet = spreadsheet.getSheets()[0];
  var sheet_name = sheet.getSheetName();
  var ep_id = sheet_name.split('_')[4]
  
  var dup_sheet = spreadsheet.duplicateActiveSheet();
  dup_sheet.setName("<name>_sd_"+ep_id);
  ();
  var protection = spreadsheet.getActiveSheet().protect();
  protection.addEditors(['<email_id>']);
};


function onEdit(e) {
  var sheet = e.source.getActiveSheet();
  var editedColumn = e.range.getColumn();
  var is_CW_Col = 3; // Column C

  if (editedColumn == is_CW_Col) {
    var rowIndex = e.range.getRow();
    var is_CW = e.value;
    var rel_lvl_col = 4; // Column D 

    if (is_CW == "FALSE") {
      var rule_false = SpreadsheetApp.newDataValidation().requireTextEqualTo("").build();
      sheet.getRange(rowIndex, rel_lvl_col).setDataValidation(rule_false);
      sheet.getRange(rowIndex, rel_lvl_col).setValue("");
    } else if (is_CW == "TRUE") {
      var rule_true = SpreadsheetApp.newDataValidation().requireValueInList(['High', 'Medium', 'Low'], true).build();
      sheet.getRange(rowIndex, rel_lvl_col).setDataValidation(rule_true);
      sheet.getRange(rowIndex, rel_lvl_col).setValue("#CHOOSE#");
    }
  }
}
