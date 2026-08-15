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
        self.tok = load_tokenizer(repo_id=tokenizer_repo or model_repo, revision=revision, allow_local_fallback=False)
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
    model = ModelInference(model_repo="sullivan1502/base-action-sft")
    
    prompt = "<chart> <O_844> <H_880> <L_844> <C_862> <O_862> <H_869> <L_816> <C_816> <O_816> <H_827> <L_814> <C_817> <O_817> <H_817> <L_794> <C_800> <O_800> <H_812> <L_786> <C_788> <O_789> <H_789> <L_732> <C_741> <O_741> <H_744> <L_728> <C_738> <O_742> <H_767> <L_713> <C_734> <O_741> <H_745> <L_736> <C_743> <O_744> <H_787> <L_744> <C_786> <O_786> <H_826> <L_751> <C_821> <O_816> <H_858> <L_764> <C_850> <O_853> <H_954> <L_843> <C_932> <O_933> <H_957> <L_875> <C_898> <O_890> <H_959> <L_890> <C_958> <O_958> <H_958> <L_884> <C_900> <O_901> <H_932> <L_900> <C_913> <O_913> <H_937> <L_900> <C_925> <O_924> <H_924> <L_879> <C_881> <O_881> <H_881> <L_853> <C_853> <O_853> <H_899> <L_853> <C_879> <O_879> <H_896> <L_838> <C_855> <O_855> <H_863> <L_836> <C_838> <O_838> <H_844> <L_824> <C_838> <O_838> <H_854> <L_793> <C_813> <O_813> <H_838> <L_723> <C_745> <O_745> <H_757> <L_623> <C_659> <O_660> <H_669> <L_584> <C_652> <O_649> <H_667> <L_628> <C_635> <O_635> <H_635> <L_576> <C_586> <O_586> <H_607> <L_438> <C_531> <O_531> <H_533> <L_484> <C_503> <O_502> <H_511> <L_447> <C_463> <O_457> <H_509> <L_425> <C_461> <O_461> <H_565> <L_460> <C_557> <O_560> <H_621> <L_517> <C_621> <O_621> <H_694> <L_621> <C_675> <O_675> <H_763> <L_646> <C_750> <O_750> <H_786> <L_689> <C_770> <O_770> <H_778> <L_739> <C_770> <O_769> <H_777> <L_737> <C_751> <O_752> <H_771> <L_732> <C_733> <O_734> <H_738> <L_700> <C_737> <O_746> <H_771> <L_714> <C_771> <O_771> <H_808> <L_745> <C_758> <O_759> <H_842> <L_758> <C_826> <O_831> <H_898> <L_824> <C_896> <O_896> <H_937> <L_847> <C_847> <O_847> <H_984> <L_846> <C_976> <O_976> <H_993> <L_923> <C_976> <O_975> <H_991> <L_943> <C_943> <O_943> <H_963> <L_914> <C_940> <O_940> <H_972> <L_938> <C_949> <O_947> <H_991> <L_946> <C_986> <O_986> <H_1056> <L_960> <C_1044> <O_1048> <H_1135> <L_1042> <C_1119> <O_1118> <H_1148> <L_1053> <C_1120> <O_1120> <H_1130> <L_1091> <C_1114> <O_1114> <H_1169> <L_1039> <C_1041> <O_1041> <H_1087> <L_1038> <C_1070> <O_1069> <H_1086> <L_1038> <C_1045> <O_1045> <H_1083> <L_1044> <C_1053> <O_1055> <H_1090> <L_1048> <C_1048> <O_1047> <H_1120> <L_1037> <C_1102> <O_1102> <H_1123> <L_1089> <C_1120> <O_1120> <H_1144> <L_1110> <C_1114> <O_1114> <H_1142> <L_1093> <C_1134> <O_1134> <H_1185> <L_1088> <C_1096> <O_1095> <H_1159> <L_1093> <C_1128> <O_1145> <H_1186> <L_1141> <C_1182> <O_1176> <H_1214> <L_1125> <C_1193> <O_1195> <H_1214> <L_1176> <C_1204> <O_1204> <H_1214> <L_1177> <C_1193> <O_1192> <H_1198> <L_1143> <C_1143> <O_1143> <H_1149> <L_1094> <C_1106> <O_1106> <H_1110> <L_1067> <C_1093> <O_1092> <H_1097> <L_1052> <C_1073> <O_1073> <H_1078> <L_1024> <C_1045> <O_1045> <H_1046> <L_1001> <C_1030> <O_1029> <H_1096> <L_1022> <C_1084> <O_1083> <H_1117> <L_1065> <C_1107> <O_1108> <H_1108> <L_1020> <C_1031> <O_1028> <H_1096> <L_1016> <C_1086> <O_1089> <H_1092> <L_1067> <C_1088> <O_1088> <H_1155> <L_1084> <C_1137> <O_1143> <H_1195> <L_1133> <C_1157> <O_1157> <H_1186> <L_1136> <C_1175> <O_1174> <H_1196> <L_1164> <C_1188> <O_1188> <H_1326> <L_1188> <C_1321> <O_1322> <H_1445> <L_1306> <C_1428> <O_1427> <H_1447> <L_1383> <C_1408> <O_1407> <H_1492> <L_1407> <C_1482> <O_1482> <H_1483> <L_1446> <C_1482> <O_1482> <H_1530> <L_1442> <C_1516> <O_1519> <H_1621> <L_1517> <C_1615> <O_1613> <H_1622> <L_1564> <C_1564> <O_1567> <H_1617> <L_1548> <C_1604> <O_1604> <H_1605> <L_1514> <C_1527> <O_1527> <H_1555> <L_1513> <C_1539> <O_1539> <H_1572> <L_1505> <C_1570> </chart> <think> <trend>DOWN</trend> <current_price> 1 5 7 0 </current_price> <zone_resistance> 1 5 8 6 : 1 6 5 0 </zone_resistance> </think>"
    completions = model.generate_one(prompt, n_gen=16)
    for completion in completions:
        print(completion)