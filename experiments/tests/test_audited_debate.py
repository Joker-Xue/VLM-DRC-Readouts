import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.agents.audited_debate import stop_action, visible_history, validate_output, digest, decision, system_prompt
from experiments.eval.eval_corrected_debate import extract, unique_messages

class ProtocolTests(unittest.TestCase):
    def test_unanimity_and_three_votes_stop(self):
        for h,n in [(4,0),(0,4),(3,1),(1,3),(3,0)]:
            self.assertEqual(stop_action(1, {'hotspot':h,'not_hotspot':n}), 'stop')
    def test_reversal_precedes_stop(self):
        self.assertEqual(stop_action(2, {'hotspot':3,'not_hotspot':1}, {'hotspot':1,'not_hotspot':2}), 'continue')
        self.assertEqual(stop_action(2, {'hotspot':0,'not_hotspot':4}, {'hotspot':2,'not_hotspot':1}), 'continue')
    def test_tie_is_not_reversal(self):
        self.assertEqual(stop_action(2, {'hotspot':3,'not_hotspot':1}, {'hotspot':2,'not_hotspot':2}), 'stop')
    def test_final_tie_and_single_vote_abstain(self):
        self.assertEqual(decision({'hotspot':2,'not_hotspot':2}), 'abstain')
        self.assertEqual(stop_action(3, {'hotspot':1,'not_hotspot':0}), 'abstain')
    def test_barrier_and_own_history(self):
        history=[{'round':r,'agent':a} for r in (1,2,3) for a in ('a','b')]
        self.assertEqual(visible_history(history,'a',1,True), [])
        self.assertEqual(len(visible_history(history,'a',2,True)),2)
        self.assertEqual(visible_history(history,'a',2,False),[{'round':1,'agent':'a'}])
    def test_no_fabrication_and_context_validation(self):
        m={'vote':'abstain','p_hotspot':.5,'rationale':'uncertain','depends_on_agents':[]}
        self.assertEqual(validate_output(m,'a',1,('a','b')),[])
        self.assertEqual(m['vote'],'abstain')
        self.assertEqual(m['depends_on_agents'],[])
        self.assertTrue(validate_output({**m,'agent':'a'},'a',1,('a','b')))
        self.assertTrue(validate_output({**m,'depends_on_agents':['b']},'a',1,('a','b')))
        self.assertTrue(validate_output({**m,'p_hotspot':[.1]},'a',1,('a','b')))
        self.assertTrue(validate_output({**m,'p_hotspot':float('nan')},'a',1,('a','b')))
    def test_cache_identity(self):
        base={'image':'abc','temperature':.7,'model':'xyz','seed':2,'max_tokens':512,'max_attempts':3}
        for field,val in [('image','def'),('temperature',.3),('model','other'),('seed',3),('max_tokens',256),('max_attempts',2)]:
            self.assertNotEqual(digest(base),digest({**base,field:val}))

    def test_prompt_forbids_self_attribution_and_uses_explicit_allowlist(self):
        prompt = system_prompt('layout_arbiter', 'layout_arbiter')
        self.assertIn('allowed_depends_on_agents', prompt)
        self.assertIn('never include your own agent name', prompt)

    def test_corrected_result_extraction_does_not_duplicate_shared_round_one(self):
        def round_messages(rnd, prefix):
            return [
                {'agent': f'a{i}', 'round': rnd, 'call_records': [f'{prefix}-{rnd}-{i}']}
                for i in range(4)
            ]
        r1 = round_messages(1, 'shared')
        configuration = {
            'single': {'score': .1, 'decision': 'not_hotspot'},
            'r1': {'score': .2, 'decision': 'not_hotspot'},
            'branches': {
                'own': {'rounds': [r1, round_messages(2, 'own'), round_messages(3, 'own')],
                        'fixed': {'score': .3, 'decision': 'not_hotspot'},
                        'early': {'score': .4, 'decision': 'not_hotspot'}},
                'peer': {'rounds': [r1, round_messages(2, 'peer'), round_messages(3, 'peer')],
                         'fixed': {'score': .5, 'decision': 'hotspot'},
                         'early': {'score': .6, 'decision': 'hotspot'}},
            },
        }
        self.assertEqual(len(unique_messages(configuration)), 20)
        self.assertEqual(extract(configuration, 'peer_fixed')['score'], .5)

if __name__=='__main__': unittest.main()
