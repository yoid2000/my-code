Use this as your `sbatch` file. It handles exactly the timeout ladder

`5 minutes -> 30 minutes -> 4 hours -> 1 day -> 7 days`

and, after each timeout, resubmits the same `job_num` with a delay equal to **half of the next timeout**:

`15 minutes, 2 hours, 12 hours, 3 days 12 hours`.

It works both for a single job number and for a Slurm array task. It uses an early batch-shell signal plus `squeue`’s `EndTime` so the restart is scheduled relative to the job’s expected end time, not relative to when the warning signal arrives. That avoids restarting too early. Slurm supports `--signal=B:...`, `--begin=...`, and `squeue -O EndTime`; command-line options also override `#SBATCH` lines, which is how the resubmitted jobs get the longer time limits. ([Slurm][1])

One implementation detail matters: shells often do not process a trapped signal while a foreground child is running, so the script launches `python program.py ...` in the background and `wait`s on it. That is the reliable way to let the trap run before Slurm kills the job. Job arrays are supported with `--array` if you want to launch many indices at once. ([Slurm][2])

```bash
#!/bin/bash
#SBATCH --job-name=adaptive_py
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm-%x-%j.out

# Fill these in for your cluster if needed:
##SBATCH --partition=your_partition
##SBATCH --account=your_account

set -u

# Timeout ladder
TIME_LIMITS=(
  "00:05:00"    # 5 minutes
  "00:30:00"    # 30 minutes
  "04:00:00"    # 4 hours
  "1-00:00:00"  # 1 day
  "7-00:00:00"  # 7 days
)

# Delay before resubmitting to the NEXT stage, in seconds:
# after 5m  -> wait 15m
# after 30m -> wait 2h
# after 4h  -> wait 12h
# after 1d  -> wait 3d12h
DELAY_SECONDS=(
  900
  7200
  43200
  302400
)

# Retry stage comes from the environment on resubmission.
STAGE="${RETRY_STAGE:-0}"

# Support either:
#   sbatch adaptive_retry.sbatch 123
# or
#   sbatch --array=0-39999 adaptive_retry.sbatch
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "4294967294" ]]; then
  JOB_NUM="${SLURM_ARRAY_TASK_ID}"
  ARRAY_MODE=1
else
  JOB_NUM="${1:?usage: sbatch adaptive_retry.sbatch <job_num>}"
  ARRAY_MODE=0
fi

if (( STAGE < 0 || STAGE >= ${#TIME_LIMITS[@]} )); then
  echo "Invalid RETRY_STAGE=${STAGE}" >&2
  exit 2
fi

# Make a stable copy of this script in the submit directory so resubmissions
# do not depend on Slurm's internal spool copy of the batch script.
SELF="${SELF:-${SLURM_SUBMIT_DIR}/.adaptive_retry.sbatch}"
if [[ ! -f "${SELF}" ]]; then
  cp "$0" "${SELF}"
  chmod 700 "${SELF}"
fi

resubmit_next_stage() {
  local next_stage=$((STAGE + 1))

  if (( next_stage >= ${#TIME_LIMITS[@]} )); then
    echo "[$(date)] JOB_NUM=${JOB_NUM}: reached max timeout (${TIME_LIMITS[$STAGE]}). No more resubmissions."
    return 0
  fi

  local expected_end
  expected_end="$(squeue -h -j "${SLURM_JOB_ID}" -O EndTime | head -n1 | xargs)"

  if [[ -z "${expected_end}" || "${expected_end}" == "N/A" || "${expected_end}" == "Unknown" ]]; then
    echo "[$(date)] Could not read EndTime from squeue; falling back to current time." >&2
    expected_end="$(date '+%Y-%m-%dT%H:%M:%S')"
  fi

  local begin_time
  begin_time="$(date -d "${expected_end} + ${DELAY_SECONDS[$STAGE]} seconds" '+%Y-%m-%dT%H:%M:%S')"

  local next_limit="${TIME_LIMITS[$next_stage]}"

  echo "[$(date)] JOB_NUM=${JOB_NUM}: timed out at stage ${STAGE} (${TIME_LIMITS[$STAGE]})."
  echo "[$(date)] Resubmitting for stage ${next_stage} with --time=${next_limit} --begin=${begin_time}"

  if (( ARRAY_MODE == 1 )); then
    sbatch \
      --array="${SLURM_ARRAY_TASK_ID}" \
      --time="${next_limit}" \
      --begin="${begin_time}" \
      --export=ALL,RETRY_STAGE="${next_stage}",SELF="${SELF}" \
      "${SELF}"
  else
    sbatch \
      --time="${next_limit}" \
      --begin="${begin_time}" \
      --export=ALL,RETRY_STAGE="${next_stage}",SELF="${SELF}" \
      "${SELF}" "${JOB_NUM}"
  fi
}

timeout_handler() {
  trap '' USR1

  echo "[$(date)] JOB_NUM=${JOB_NUM}: received timeout warning signal at stage ${STAGE}."

  if [[ -n "${child_pid:-}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    resubmit_next_stage

    # Ask the child to stop cleanly; if it does not, Slurm will kill it at the job limit.
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  else
    echo "[$(date)] JOB_NUM=${JOB_NUM}: child already finished; no resubmission needed."
  fi

  exit 0
}

trap timeout_handler USR1

echo "[$(date)] Starting JOB_NUM=${JOB_NUM}, STAGE=${STAGE}, LIMIT=${TIME_LIMITS[$STAGE]}, SLURM_JOB_ID=${SLURM_JOB_ID}"

python program.py "${JOB_NUM}" &
child_pid=$!

wait "${child_pid}"
rc=$?

if (( rc == 0 )); then
  echo "[$(date)] JOB_NUM=${JOB_NUM}: finished successfully."
  exit 0
fi

echo "[$(date)] JOB_NUM=${JOB_NUM}: program exited with status ${rc} before timeout handling. Not auto-resubmitting."
exit "${rc}"
```

Submit a single job like this:

```bash
sbatch adaptive_retry.sbatch 123
```

Submit all ~40,000 as an array like this, with a cap on concurrent tasks:

```bash
sbatch --array=0-39999%500 adaptive_retry.sbatch
```

Two caveats. First, `--signal` can arrive up to 60 seconds earlier than requested, and `--begin` scheduling is only as precise as the scheduler poll interval. Second, this script **restarts from scratch** each time; it does not checkpoint partial progress. ([Slurm][1])

[1]: https://slurm.schedmd.com/sbatch.html "Slurm Workload Manager - sbatch"
[2]: https://slurm.schedmd.com/scancel.html "Slurm Workload Manager - scancel"
