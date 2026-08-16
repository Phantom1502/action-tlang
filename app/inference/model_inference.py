from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import LlamaForCausalLM

from app.tokenizer import load_tokenizer

class ModelInference:
    def __init__(
        self,
        model_repo: str,
        revision: Optional[str] = None,
        subfolder: Optional[str] = None,
        tokenizer_repo: Optional[str] = None,
        max_new_tokens: int = 24,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- Tokenizer: pin cùng revision với model nếu có, giữ quirk
        # add_eos_token=False/add_bos_token=True/padding_side=left (batch generate). ---
        self.tok = load_tokenizer(repo_id=tokenizer_repo or model_repo, revision=revision)
        self.tok.add_eos_token = False
        self.tok.add_bos_token = True
        self.tok.padding_side = "left"
        
        model_kwargs: Dict[str, Any] = {}
        if revision is not None:
            model_kwargs["revision"] = revision
        if subfolder is not None:
            model_kwargs["subfolder"] = subfolder
        self.model = LlamaForCausalLM.from_pretrained(model_repo, **model_kwargs).to(self.device)
        self.model.eval()
        if self.model.config.vocab_size != self.tok.vocab_size:
            raise ValueError(
                f"model.vocab_size ({self.model.config.vocab_size}) != tokenizer.vocab_size "
                f"({self.tok.vocab_size}) — checkpoint và tokenizer không khớp "
                f"(model_repo={model_repo!r}, revision={revision!r}, subfolder={subfolder!r})."
            )
            
    def generate_batch(self, rows: Sequence[Dict[str, Any]]) -> List[str]:
        prompts = [r["prompt"] for r in rows]
        enc = self.tok(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        if self.do_sample:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

        gen_ids = out_ids[:, input_ids.shape[1]:]
        return self.tok.batch_decode(gen_ids, skip_special_tokens=True)
    
    def generate_one(self, prompt: str, n_gen: int = 1) -> List[str]:
        prompts = [prompt] * n_gen
        return self.generate_batch([{"prompt": p} for p in prompts])
    
if __name__ == "__main__":
    model = ModelInference(model_repo="sullivan1502/base-action-grpo")
    
    prompt = "<chart> <O_707> <H_711> <L_671> <C_687> <O_687> <H_687> <L_654> <C_676> <O_675> <H_724> <L_670> <C_715> <O_715> <H_739> <L_701> <C_732> <O_733> <H_733> <L_668> <C_676> <O_674> <H_724> <L_665> <C_716> <O_719> <H_721> <L_702> <C_718> <O_718> <H_767> <L_715> <C_754> <O_758> <H_796> <L_751> <C_768> <O_768> <H_790> <L_753> <C_781> <O_781> <H_797> <L_773> <C_791> <O_791> <H_892> <L_791> <C_888> <O_889> <H_979> <L_877> <C_966> <O_966> <H_981> <L_934> <C_952> <O_951> <H_1013> <L_951> <C_1006> <O_1006> <H_1007> <L_980> <C_1006> <O_1006> <H_1041> <L_977> <C_1031> <O_1033> <H_1108> <L_1032> <C_1103> <O_1102> <H_1108> <L_1066> <C_1066> <O_1068> <H_1105> <L_1054> <C_1095> <O_1095> <H_1096> <L_1029> <C_1039> <O_1039> <H_1059> <L_1029> <C_1047> <O_1048> <H_1072> <L_1023> <C_1070> <O_1073> <H_1106> <L_1064> <C_1097> <O_1097> <H_1148> <L_1092> <C_1127> <O_1126> <H_1144> <L_1101> <C_1136> <O_1136> <H_1159> <L_1087> <C_1111> <O_1110> <H_1112> <L_1060> <C_1073> <O_1073> <H_1084> <L_1040> <C_1047> <O_1047> <H_1077> <L_1039> <C_1050> <O_1050> <H_1082> <L_1037> <C_1060> <O_1061> <H_1130> <L_1061> <C_1097> <O_1097> <H_1136> <L_1076> <C_1134> <O_1134> <H_1214> <L_1129> <C_1203> <O_1202> <H_1314> <L_1201> <C_1282> <O_1281> <H_1307> <L_1225> <C_1244> <O_1245> <H_1382> <L_1240> <C_1382> <O_1382> <H_1389> <L_1362> <C_1377> <O_1377> <H_1393> <L_1319> <C_1325> <O_1326> <H_1346> <L_1260> <C_1264> <O_1265> <H_1292> <L_1222> <C_1280> <O_1280> <H_1315> <L_1256> <C_1305> <O_1306> <H_1335> <L_1241> <C_1254> <O_1259> <H_1259> <L_1193> <C_1209> <O_1209> <H_1353> <L_1199> <C_1306> <O_1306> <H_1345> <L_1300> <C_1315> <O_1315> <H_1349> <L_1285> <C_1327> <O_1325> <H_1329> <L_1304> <C_1315> <O_1315> <H_1352> <L_1297> <C_1313> <O_1313> <H_1321> <L_1284> <C_1291> <O_1292> <H_1292> <L_1245> <C_1247> <O_1247> <H_1259> <L_1230> <C_1242> <O_1242> <H_1285> <L_1237> <C_1271> <O_1271> <H_1290> <L_1204> <C_1213> <O_1213> <H_1224> <L_1170> <C_1190> <O_1190> <H_1198> <L_1131> <C_1155> <O_1158> <H_1162> <L_1133> <C_1142> <O_1141> <H_1185> <L_1141> <C_1177> <O_1177> <H_1190> <L_1146> <C_1147> <O_1148> <H_1227> <L_1148> <C_1202> <O_1203> <H_1231> <L_1196> <C_1200> <O_1198> <H_1230> <L_1196> <C_1223> <O_1223> <H_1228> <L_1196> <C_1205> <O_1203> <H_1226> <L_1194> <C_1206> <O_1206> <H_1211> <L_1195> <C_1200> <O_1200> <H_1203> <L_1167> <C_1169> <O_1168> <H_1177> <L_1131> <C_1158> <O_1158> <H_1163> <L_1137> <C_1143> <O_1143> <H_1166> <L_1108> <C_1114> <O_1114> <H_1135> <L_1111> <C_1123> <O_1120> <H_1129> <L_1090> <C_1118> <O_1118> <H_1147> <L_1118> <C_1132> <O_1132> <H_1132> <L_1096> <C_1124> <O_1125> <H_1135> <L_1118> <C_1126> <O_1125> <H_1134> <L_1108> <C_1108> <O_1108> <H_1121> <L_1104> <C_1114> <O_1113> <H_1121> <L_1092> <C_1110> <O_1103> <H_1103> <L_1078> <C_1100> <O_1100> <H_1105> <L_1062> <C_1067> <O_1069> <H_1079> <L_1046> <C_1062> <O_1062> <H_1076> <L_1053> <C_1058> <O_1058> <H_1098> <L_1058> <C_1088> <O_1088> <H_1092> <L_1073> <C_1080> <O_1080> <H_1090> <L_1055> <C_1077> <O_1077> <H_1109> <L_1060> <C_1068> <O_1067> <H_1117> <L_1063> <C_1109> <O_1110> <H_1113> <L_1065> <C_1083> <O_1084> <H_1086> <L_1056> <C_1068> <O_1068> <H_1073> <L_1050> <C_1061> <O_1061> <H_1081> <L_1060> <C_1074> <O_1074> <H_1119> <L_1061> <C_1118> <O_1117> <H_1118> <L_1095> <C_1116> <O_1115> <H_1132> <L_1107> <C_1112> <O_1112> <H_1113> <L_1065> <C_1068> <O_1067> <H_1094> <L_1050> <C_1082> <O_1083> <H_1107> <L_1069> <C_1102> <O_1102> <H_1110> <L_1068> <C_1082> <O_1087> <H_1123> <L_1085> <C_1121> <O_1121> <H_1130> <L_1104> <C_1108> <O_1108> <H_1112> <L_1025> <C_1058> </chart> <think> <trend>RANGE</trend> <current_price> 1 0 5 8 </current_price> <zone_resistance> 1 0 5 6 : 1 1 0 8 </zone_resistance> </think>"
    completions = model.generate_one(prompt, n_gen=16)
    
    print("\n1 Sample with Zone Score: 4.716981\n")
    for completion in completions:
        print(completion)