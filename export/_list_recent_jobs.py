"""Diagnostico temporario: lista os jobs mais recentes da conta AI Hub."""

import qai_hub as hub

try:
    jobs = hub.get_job_summaries()
except Exception as exc:  # noqa: BLE001
    print("get_job_summaries falhou:", exc)
    raise SystemExit(1)

print(f"total jobs: {len(jobs)}")
for job in jobs[:14]:
    code = getattr(job.status, "code", "?")
    jtype = getattr(job.job_type, "name", "?")
    print(f"{job.date} | {job.job_id} | {jtype:10s} | {str(code):10s} | {str(job.name)[:55]}")

