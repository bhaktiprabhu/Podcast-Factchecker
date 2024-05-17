/** @OnlyCurrentDoc */

function FormattingandProtectionforOriginalSheet() {
  var spreadsheet = SpreadsheetApp.getActive();
  var sheet = spreadsheet.getActiveSheet();
  
  sheet.setRowHeight(1, 28)
  .setFrozenRows(1);
  
  sheet.getRange('A1:E1').setFontColor('#ffffff')
  .setBackground('#a64d79')
  .setFontWeight('bold')
  .setFontSize(12)
  .setHorizontalAlignment('center');
  
  sheet.getRange(1, 1, sheet.getMaxRows(), sheet.getMaxColumns()).activate();
  sheet.getActiveRangeList().setWrapStrategy(SpreadsheetApp.WrapStrategy.WRAP);
  
  sheet.setColumnWidth(1, 350)
  .setColumnWidth(2, 550)
  .setColumnWidth(3, 200)
  .setColumnWidth(4, 170)
  .setColumnWidth(5, 130);
  
  sheet.getRange('D1').setNote("Set the value to TRUE if you think that the coreference for the 'check_worthy_claim' is not resolved");
  sheet.getRange('E1').setNote("Select whether the evidence_snippet SUPPORTS or REFUTES the corresponding 'check_worthy_claim'");
  
  sheet.getRange(2, 4, sheet.getLastRow()-1).activate();
  sheet.getActiveRange().setDataValidation(SpreadsheetApp.newDataValidation()
  .setAllowInvalid(false)
  .requireValueInList(['FALSE', 'TRUE'], true)
  .build());
  
  sheet.getRange(2, 5, sheet.getLastRow()-1).activate();
  sheet.getActiveRange().setValue("#CHOOSE#");
  sheet.getActiveRange().setDataValidation(SpreadsheetApp.newDataValidation()
  .setAllowInvalid(false)
  .requireValueInList(['SUPPORTS', 'REFUTES'], true)
  .build());
  
  protection = sheet.protect();
};


function ProtectFieldsInDuplicates() {
  var spreadsheet = SpreadsheetApp.getActive();
  var protection = spreadsheet.getRange('1:1').protect().setDescription('Protect Header Row');
  protection.removeEditors(protection.getEditors());

  protection = spreadsheet.getRange('A:C').protect().setDescription('Protect Header columns');
  protection.removeEditors(protection.getEditors());
};


function MakeDuplicateSheets() {
  var spreadsheet = SpreadsheetApp.getActive();
  var sheet = spreadsheet.getSheets()[0];
  var sheet_name = sheet.getSheetName();
  var ep_id = sheet_name.split('_')[4]
  
  var dup_sheet = spreadsheet.duplicateActiveSheet();
  dup_sheet.setName("<name>_sd_"+ep_id);
  ProtectFieldsInDuplicates();
  var protection = spreadsheet.getActiveSheet().protect();
  protection.addEditors(['<email_id>']);
};