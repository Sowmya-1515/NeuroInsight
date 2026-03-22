path = '/Users/sowmyaalamuri/Desktop/Capstone_project/Phase2/step2_model.py'

with open(path, 'r') as f:
    lines = f.readlines()

# Find the class MultiTaskLoss line
start = None
for i, line in enumerate(lines):
    if 'class MultiTaskLoss(nn.Module):' in line:
        start = i
        break

if start is None:
    print('ERROR: Could not find MultiTaskLoss class')
    exit(1)

print(f'Found MultiTaskLoss at line {start + 1}')

# Build the new class definition
new_class = '''class MultiTaskLoss(nn.Module):
    def __init__(self, grade_weight=None, severity_weight=None,
                 size_weight=None, location_weight=None):
        super().__init__()
        self.supcon_loss = losses.SupConLoss(temperature=0.07)
        self.ntxent_loss = losses.NTXentLoss(temperature=0.07)
        self.ce_grade    = nn.CrossEntropyLoss(weight=grade_weight)
        self.ce_severity = nn.CrossEntropyLoss(weight=severity_weight)
        self.ce_size     = nn.CrossEntropyLoss(weight=size_weight)
        self.ce_location = nn.CrossEntropyLoss(weight=location_weight)
        self.w_supcon    = 1.0
        self.w_ntxent    = 0.5
        self.w_grade     = 0.3
        self.w_severity  = 0.3
        self.w_size      = 0.2
        self.w_location  = 0.2
    def forward(self, outputs, batch):
        emb  = outputs["embedding"]
        rlbl = batch["retrieval_label"]
        supcon = self.supcon_loss(emb, rlbl)
        ntxent = self.ntxent_loss(emb, rlbl)
        grade_ce    = self.ce_grade(outputs["grade_logits"],       batch["grade_label"])
        severity_ce = self.ce_severity(outputs["severity_logits"], batch["severity_label"])
        size_ce     = self.ce_size(outputs["size_logits"],         batch["size_label"])
        location_ce = self.ce_location(outputs["location_logits"], batch["location_label"])
        total = (self.w_supcon   * supcon      +
                 self.w_ntxent   * ntxent      +
                 self.w_grade    * grade_ce    +
                 self.w_severity * severity_ce +
                 self.w_size     * size_ce     +
                 self.w_location * location_ce)
        return {
            "total":    total,
            "supcon":   supcon.item(),
            "ntxent":   ntxent.item(),
            "grade":    grade_ce.item(),
            "severity": severity_ce.item(),
            "size":     size_ce.item(),
            "location": location_ce.item()
        }
'''

# Find where the class ends (next class definition or end of file)
end = len(lines)
for i in range(start + 1, len(lines)):
    if lines[i].startswith('class '):
        end = i
        break

print(f'Replacing lines {start + 1} to {end}')

# Replace the old class with new
new_lines = lines[:start] + [new_class] + lines[end:]

with open(path, 'w') as f:
    f.writelines(new_lines)

print('✓ step2_model.py patched successfully!')
print()
print('Verifying...')
with open(path, 'r') as f:
    content = f.read()
if 'grade_weight=None' in content and 'ce_grade' in content:
    print('✓ Verification passed — class weights are in place!')
else:
    print('✗ Verification failed — please check manually')