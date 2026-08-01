/* Leaderboard: loads data/leaderboard.json, renders the table, and wires the
 * agent-class filter tabs. No dependencies.
 */
(() => {
  const CLASS_LABELS = {
    A: "A · Text",
    B: "B · Voice (audio-native)",
    C: "C · Voice (pipeline)"
  };
  const PASS_KS = [1, 2, 3, 4];

  let rankedRows = [];

  function fmtPass(row, k) {
    const v = row.pass_hat && row.pass_hat[String(k)];
    if (typeof v !== "number") return "—";
    return v.toFixed(3);
  }

  function esc(s) {
    const div = document.createElement("div");
    div.textContent = s == null ? "—" : String(s);
    return div.innerHTML;
  }

  function render(filterClass) {
    const tbody = document.getElementById("leaderboard-body");
    const rows =
      filterClass === "all"
        ? rankedRows
        : rankedRows.filter((r) => r.class === filterClass);

    if (rows.length === 0) {
      tbody.innerHTML =
        '<tr><td colspan="11" class="empty-state">No results yet for this agent class.</td></tr>';
      return;
    }

    tbody.innerHTML = rows
      .map((r) => {
        const badge = r.preliminary
          ? ' <span class="badge">preliminary · ' + esc(r.split) + " split</span>†"
          : "";
        const trials =
          esc(r.trials) + (r.split && r.split !== "full" ? " (" + esc(r.split) + ")" : "");
        return (
          "<tr>" +
          '<td class="num">' + r.rank + "</td>" +
          "<td>" + esc(CLASS_LABELS[r.class] || r.class) + "</td>" +
          "<td>" + esc(r.model || r.agent) + badge + "</td>" +
          "<td>" + esc(r.user_sim) + "</td>" +
          "<td>" + esc(r.judge) + "</td>" +
          '<td class="num">' + trials + "</td>" +
          PASS_KS.map((k) => '<td class="num">' + fmtPass(r, k) + "</td>").join("") +
          "<td>" + esc(r.date) + "</td>" +
          "</tr>"
        );
      })
      .join("");
  }

  function initTabs() {
    const tabs = document.querySelectorAll(".filter-tabs button");
    tabs.forEach((btn) => {
      btn.addEventListener("click", () => {
        tabs.forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
        render(btn.dataset.class);
      });
    });
  }

  function showError() {
    document.getElementById("leaderboard-body").innerHTML =
      '<tr><td colspan="11" class="empty-state">Could not load ' +
      "<code>data/leaderboard.json</code>. If you opened this page from " +
      "<code>file://</code>, serve the directory instead (browsers block " +
      "local <code>fetch</code>), e.g. <code>python3 -m http.server</code>." +
      "</td></tr>";
  }

  document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    fetch("data/leaderboard.json")
      .then((res) => {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then((data) => {
        // Global rank by pass^1 (desc), tie-broken by pass^2 then pass^3.
        rankedRows = (data.rows || [])
          .slice()
          .sort((a, b) => {
            for (const k of PASS_KS) {
              const av = (a.pass_hat && a.pass_hat[String(k)]) || 0;
              const bv = (b.pass_hat && b.pass_hat[String(k)]) || 0;
              if (bv !== av) return bv - av;
            }
            return 0;
          })
          .map((row, i) => ({ ...row, rank: i + 1 }));
        render("all");
      })
      .catch(showError);
  });
})();
