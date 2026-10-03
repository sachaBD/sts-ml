2026-10-02T23:29:56Z START ah-c01 ckpt=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt seeds=971000000000+2000 eps=0.05 route=0.15 extra=--key-rule
SRE LAUNCH 2026-10-03T00:29:57+01:00 pid=1276888 launch1.sh (c01,c02 ~1.5-2h each, dev-inc ~30-45m; total ~4-5h) log=experiments/act3-heart/launch1.log
2026-10-03T02:02:44+01:00 SRE: ah-c01 failed 1998/2000 (seeds 971000000292, 971000001879: combat_v4 rebuild mismatch); chain halted; lead notified
2026-10-03T01:04:57Z START train ah-r0a-v1init --init runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out --lam 1 --epochs 20
2026-10-03T01:06:25Z DONE train ah-r0a-v1init
2026-10-03T01:06:25Z START train ah-r0a-v1 --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out --lam 1 --epochs 20
2026-10-03T01:08:10Z DONE train ah-r0a-v1
2026-10-03T01:08:10Z START train ah-r0a-v31 --arch-spec agents/overworld/value/architectures/rp3.1-w64-h128-l2.toml --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out --lam 1 --epochs 20
SRE LAUNCH 2026-10-03T02:18:51+01:00 pid=1279494 launch1b.sh (c02 ~80m, dev-inc ~30m) log=experiments/act3-heart/launch1b.log
2026-10-03T01:18:51Z START ah-c02 ckpt=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt seeds=971000002000+2000 eps=0.05 route=0.15 extra=--key-rule-p 0.5
2026-10-03T01:30:51Z DONE train ah-r0a-v31
2026-10-03T01:30:51Z START train ah-r0a-v32 --arch-spec agents/overworld/value/architectures/rp3.2-w64-h128-l2.toml --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out --lam 1 --epochs 20
2026-10-03T02:01:08Z DONE train ah-r0a-v32
2026-10-03T02:51:52Z DONE ah-c02
2026-10-03T02:51:52Z START ah-dev-inc ckpt=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt seeds=961000000000+600 eps=0 route=0 extra=--key-rule
2026-10-03T02:54:04Z START train ah-r0b-v1init --init runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out --lam 1 --epochs 20
2026-10-03T02:54:05Z START train ah-r0b-v31 --arch-spec agents/overworld/value/architectures/rp3.1-w64-h128-l2.toml --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out --lam 1 --epochs 20
2026-10-03T02:54:05Z START train ah-r0b-v32 --arch-spec agents/overworld/value/architectures/rp3.2-w64-h128-l2.toml --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out --lam 1 --epochs 20
2026-10-03T03:00:37Z DONE train ah-r0b-v1init
2026-10-03T03:00:37Z START train ah-r0b-v1td --init runs/schema=value_net_v1/date=2026-10-03/id=ah-r0a-v1init/out/model.pt --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out --lam .7 --epochs 20
2026-10-03T03:05:35Z DONE train ah-r0b-v1td
2026-10-03T03:54:53Z DONE ah-dev-inc
SRE LAUNCH 2026-10-03T05:04:13+01:00 launch2.sh (ah-dev-r0a-v1init ~35m) log=experiments/act3-heart/launch2.log
2026-10-03T04:04:13Z START ah-dev-r0b-v1td ckpt=runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt seeds=961000000000+600 eps=0 route=0 extra=--key-rule
2026-10-03T04:17:57Z DONE train ah-r0b-v32
2026-10-03T04:51:40Z LEAD LAUNCH launch3.sh (ah-dev-r0b-v32, ~45 min)
2026-10-03T04:51:40Z START ah-dev-r0b-v32 ckpt=runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v32/out/model.pt seeds=961000000000+600 eps=0 route=0 extra=--key-rule
2026-10-03T04:53:46Z DONE train ah-r0b-v31
2026-10-03T05:16:42Z DONE ah-dev-r0b-v32
2026-10-03T05:27:21Z START train ah-r0b-v1 --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out --lam 1 --epochs 20
2026-10-03T05:31:09Z DONE train ah-r0b-v1
2026-10-03T05:31:50Z START train ah-r0c-v31d --arch-spec agents/overworld/value/architectures/rp3.1-w64-h128-l2.toml --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out --lam 1 --epochs 20 --distill runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt
2026-10-03T05:31:50Z START train ah-r0c-v32d --arch-spec agents/overworld/value/architectures/rp3.2-w64-h128-l2.toml --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out --lam 1 --epochs 20 --distill runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt
2026-10-03T05:32:44Z LEAD LAUNCH launch4.sh (ah-dev-hyb ~45 min, then ah-c03 ~90 min)
2026-10-03T05:32:44Z START ah-dev-hyb ckpt=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt seeds=961000000000+600 eps=0 route=0 extra=--key-rule --late-ckpt runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt --late-act 3
2026-10-03T06:09:46Z DONE ah-dev-hyb
2026-10-03T06:09:46Z START ah-c03 ckpt=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt seeds=971000004000+2000 eps=0.05 route=0.15 extra=--key-rule-p 0.5 --late-ckpt runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt --late-act 3
2026-10-03T06:18:45Z START ah-c04 ckpt=runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt seeds=971000006000+2000 eps=0.05 route=0.15 extra=--key-rule-p 0.5
2026-10-03T06:18:46Z LEAD LAUNCH launch5.sh (ah-c04: 2000 runs with r0b-v1td, ~90 min); ah-c03 stopped at ~160 runs (kept as partial)
2026-10-03T06:31:40Z DONE train ah-r0c-v31d
2026-10-03T06:33:28Z DONE train ah-r0c-v32d
2026-10-03T06:51:14Z START train ah-r0d-v31d --arch-spec agents/overworld/value/architectures/rp3.1-w64-h128-l2.toml --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out --lam 1 --epochs 20 --distill runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt --distill-weight 2 --batch-size 128
2026-10-03T06:51:14Z START train ah-r0d-v32d --arch-spec agents/overworld/value/architectures/rp3.2-w64-h128-l2.toml --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out --lam 1 --epochs 20 --distill runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt --distill-weight 2 --batch-size 128
2026-10-03T07:50:06Z DONE train ah-r0d-v32d
2026-10-03T07:52:31Z DONE train ah-r0d-v31d
2026-10-03T07:58:48Z DONE ah-c04
2026-10-03T08:04:48Z START train ah-r1-v1td --init runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c03/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c04/out --lam .7 --decay .85 --epochs 20
2026-10-03T08:06:36Z START ah-dev-r0d-v32d ckpt=runs/schema=value_net_v1/date=2026-10-03/id=ah-r0d-v32d/out/model.pt seeds=961000000000+600 eps=0 route=0 extra=--key-rule
2026-10-03T08:06:37Z LEAD LAUNCH launch6.sh (ah-dev-r0d-v32d, ~45 min)
2026-10-03T08:09:42Z DONE train ah-r1-v1td
2026-10-03T08:09:59Z START train ah-r1-v32d --init runs/schema=value_net_v1/date=2026-10-03/id=ah-r0d-v32d/out/model.pt --data runs/schema=overworld_v1/date=2026-10-02/id=ah-c01/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c02/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c03/out runs/schema=overworld_v1/date=2026-10-03/id=ah-c04/out --lam 1 --decay .85 --epochs 20 --distill runs/schema=value_net_v1/date=2026-10-03/id=ah-r1-v1td/out/model.pt --distill-weight 2 --batch-size 128
2026-10-03T08:42:08Z DONE ah-dev-r0d-v32d
2026-10-03T08:42:30Z LEAD LAUNCH (chained) ah-dev-r1-v1td
2026-10-03T08:42:30Z START ah-dev-r1-v1td ckpt=runs/schema=value_net_v1/date=2026-10-03/id=ah-r1-v1td/out/model.pt seeds=961000000000+600 eps=0 route=0 extra=--key-rule
2026-10-03T08:46:21Z DONE train ah-r1-v32d
2026-10-03T09:09:32Z DONE ah-dev-r1-v1td
2026-10-03T09:12:57Z START ah-fresh-r0d-v32d ckpt=runs/schema=value_net_v1/date=2026-10-03/id=ah-r0d-v32d/out/model.pt seeds=981000000000+800 eps=0 route=0 extra=--key-rule
2026-10-03T09:12:58Z LEAD LAUNCH launch8.sh (fresh v32d ~45m, fresh inc ~45m, dev r1-v32d ~45m, fresh v1td ~45m)
2026-10-03T09:52:59Z DONE ah-fresh-r0d-v32d
2026-10-03T09:52:59Z START ah-fresh-inc ckpt=runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt seeds=981000000000+800 eps=0 route=0 extra=--key-rule
2026-10-03T10:31:01Z DONE ah-fresh-inc
2026-10-03T10:31:01Z START ah-dev-r1-v32d ckpt=runs/schema=value_net_v1/date=2026-10-03/id=ah-r1-v32d/out/model.pt seeds=961000000000+600 eps=0 route=0 extra=--key-rule
2026-10-03T11:06:02Z DONE ah-dev-r1-v32d
2026-10-03T11:06:02Z START ah-fresh-r0b-v1td ckpt=runs/schema=value_net_v1/date=2026-10-03/id=ah-r0b-v1td/out/model.pt seeds=981000000000+800 eps=0 route=0 extra=--key-rule
2026-10-03T11:47:35Z DONE ah-fresh-r0b-v1td
