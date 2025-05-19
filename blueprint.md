# Zero-Code User Blueprint for SPARC Program Generation

**Project Title:** ECON1500 Games  
**Prepared By:** Fernando Duarte  
**Date:** 2025-05-18  

**Instructions for You (The Visionary!):**

* **No Tech Jargon Needed!** Just describe your idea in plain English. Think about what you want the program to do and why, not how it does it technically.  
* **Be Detailed:** The more information and specific examples you give, the better the AI (our team of virtual coding assistants, called SPARC) can understand and build exactly what you want. Imagine you're describing it to someone who needs to build it perfectly without asking you follow-up questions.  
* **Focus on the Goal:** What problem does this solve? What process does it make easier?  
* **Don't Worry About Code:** SPARC will figure out the best programming languages, databases, and technical stuff based on your description and its own research.  

---

## Section 1: The Big Picture – What is this program all about?

1. **Elevator Pitch:**  
   ECON1500 Games is a browser-based economic simulation that lets students step into the role of policymakers for China’s open-economy growth model (1980–2025). Using Next.js and TypeScript, participants set saving rates and exchange-rate policies each round and watch real-time impacts on GDP, trade balances, capital, productivity, and other key indicators.

2. **Problem Solver:**  
   It replaces static lectures and spreadsheets with an interactive, multiplayer environment where students experiment with policy choices and see immediate, data-driven feedback—deepening their understanding of growth theory and international trade dynamics.

3. **Why Does This Need to Exist?**  
   - **Engagement:** Transforms abstract equations into hands-on learning.  
   - **Insight:** Helps students visualize how policy levers affect long-term economic outcomes.  
   - **Efficiency:** Automates model computation and data visualization so instructors can focus on discussion rather than setup.

---

## Section 2: The Users – Who is this program for?

1. **Primary Users:**  
   - Undergraduate and graduate economics students  
   - Economics instructors teaching growth theory or international macro  

2. **User Goals:**  
   1. **Explore Policy Impact:** Try different saving rates and exchange-rate regimes to see how they influence GDP, trade, and capital over decades.  
   2. **Collaborate & Compete:** Join classmates in multiplayer sessions, compare outcomes on a leaderboard, and discuss strategies.  
   3. **Review & Reflect:** Access a round-by-round history to understand the cause-and-effect of decisions and improve in subsequent simulations.

---

## Section 3: The Features – What can the program do?

1. **Core Actions:**  
   - Create an account & securely log in  
   - Join or create a game session for a specific class or team  
   - Choose or customize an economics-themed team name  
   - Set policy controls each round: saving rate ($s_t$) & exchange-rate factor ($x_t$)  
   - Submit decisions and view real-time updates via WebSockets  
   - Monitor economic indicators on a visual dashboard (GDP, exports/imports, TFP, capital stock)  
   - Inspect past rounds in a History view  
   - Compare performance on a live Leaderboard  
   - Adjust game settings (round length, model parameters)  
   - Download simulation data for further analysis

2. **Key Feature Deep Dive – Real-Time Dashboard Interaction:**  
   1. **Entry Point:** After logging in and joining a session, the student lands on `app/game/page.tsx`.  
   2. **Policy Panel:** On the left, they see sliders or input fields (`components/game/Controls.tsx`) for saving rate and exchange-rate policy.  
   3. **Submit & Feedback:** Hitting “Submit” sends their choices via a Socket.IO channel.  
   4. **Live Charts:** On the right, `components/game/Dashboard.tsx` updates pie charts (`components/game/GdpPieChart.tsx`), line graphs of GDP over time, and numeric indicators within seconds.  
   5. **Peer Actions:** A sidebar shows other players’ latest round decisions and a running chat for discussion.  
   6. **Next Round:** Once all players submit or the timer elapses, the engine publishes the new state; the dashboard animates to reflect new values, and controls unlock for the next round.

---

## Section 4: The Information – What does it need to handle?

1. **Information Needed:**  
   - **User Data:** Student IDs, names, login credentials  
   - **Session Data:** Game ID, roster, round timer, settings  
   - **Model Inputs:** Player controls ($s_t$, $x_t$), exogenous variables (counterfactual exchange rate, FDI ratio, foreign income, human capital index)  
   - **Model State:** Capital stock ($K_t$), labor force ($L_t$), TFP ($A_t$) each round  
   - **Economic Outcomes:** GDP ($Y_t$), exports ($X_t$), imports ($M_t$), net exports, consumption, investment, openness ratio  
   - **History Logs:** Timestamped decisions and computed outcomes  
   - **Leaderboard Metrics:** Cumulative GDP, efficiency scores, ranking  

2. **Data Relationships (Optional but helpful):**  
   - A **Game Session** groups multiple **Player** records.  
   - Each **Player** produces one **Decision** per **Round**.  
   - The **Game Engine** consumes all Decisions plus Exogenous Data to compute a single **RoundResult**.  
   - **RoundResult** drives updates to the next Round’s State Variables and populates the Dashboard.

3. **Details on Model and Data***
   - Follow `china_growth_model.md` for details on the economic model and data.
   - Feel free to modify parameters and paths of exogenous variables if they match China data better.
   - Allow user to pick either one of the three options for the nominal exchange rate policy (as in `china_growth_model.md`), or to set the nominal exchange rate directly as an additional feature.
   - Create instruction option to pick parameters and paths of exogenous variables in an intuitive and interactive way in real-time during web use by students.

## Section 5: The Look & Feel – How should it generally seem?

1. **Overall Style:**  
   - **Clean & Modern** with a subtle educational tone  
   - **Responsive Layout** that works on laptops and tablets  
   - **Vibrant Data Visualizations** with clear labels and animations  

2. **Similar Programs (Appearance):**  
   - MobLab and Capsim classroom simulations  
   - Interactive dashboards in platforms like ObservableHQ  
   - Economic experiment UIs in Virtual Lab environments  

---

## Section 6: The Platform – Where will it be used?

1. **Primary Environment:**  
   - ✔️ On a Website (Chrome, Safari, Firefox)  
   - Secondary: Mobile-friendly support via responsive design  

2. **(If Mobile App):**  
   - Not a standalone mobile app; the site is fully responsive and works offline only for static pages (not the live simulation).

---

## Section 7: The Rules & Boundaries – What are the non-negotiables?

1. **Must-Have Rules:**  
   - Users must authenticate with their course-issued credentials.  
   - Each player may submit exactly one set of policy choices per round.  
   - The economic model equations must follow the published Solow-style open-economy framework without modification.  
   - No negative capital stock: enforce a floor at zero to keep the model well-defined.

2. **Things to Avoid:**  
   - Don’t expose raw server logs or calculation code to the client.  
   - Don’t allow changing core model parameters mid-game.  
   - Don’t auto-execute arbitrary JavaScript from chat or user inputs.

---

## Section 8: Success Criteria – How do we know it’s perfect?

1. **Scenario 1:** After logging in and joining “ECON1500 – Spring 2025,” Maria sets $s_t=0.25$, $x_t=1.2$ and clicks Submit. Within 2 seconds, her dashboard updates showing new GDP and trade values.  
2. **Scenario 2:** In a five-player session, all participants submit decisions before the 90-second timer ends; the game advances to the next round automatically and broadcasts results to everyone.  
3. **Scenario 3:** At any point, Tom views the History tab and sees a table of past rounds with his own and classmates’ choices matched to computed outcomes, matching the published round-by-round data in “simulation_calculations.md.”

---

## Section 9: Inspirations & Comparisons – Learning from others

1. **Similar Programs (Functionality):**  
   - MobLab’s macroeconomic experiments  
   - Econ-ARK’s online Solow model demos  
   - SimCityEdu’s policy trade-off dashboards  

2. **Likes & Dislikes:**  
   - **Likes:** Real-time updates, clear visual feedback, multiplayer engagement  
   - **Dislikes:** Cluttered menus, slow chart rendering, opaque calculation steps  

---

## Section 10: Future Dreams (Optional) – Where could this go?

1. **Nice-to-Haves:**  
   - Allow instructors to upload custom growth-model parameters or countries  
   - Support asynchronous play with email or Slack notifications  
   - Add “what if” scenario planner for comparing two policy paths  

2. **Long-Term Vision:**  
   - A library of economy-simulation modules (emerging markets, commodity exporters)  
   - Integration with VR for immersive policy labs  
   - Machine-learning-driven hints to help students improve their strategies  

---

## Section 11: Technical Preferences (Strictly Optional!)

*Note: SPARC will choose defaults unless you have strong requirements.*

1. **Specific Programming Language?**  
   - **Next.js (React + TypeScript)** for universal SSR/CSR benefits and a strong component ecosystem.

2. **Specific Database?**  
   - **Supabase (PostgreSQL)** for easy authentication, real-time subscriptions, and data storage.

3. **Specific Cloud Provider?**  
   - **AWS** (Elastic Beanstalk or Fargate) with a managed WebSocket layer and RDS for production; Vercel for staging.

---

**Final Check:**

- ✅ All sections 1–9 answered in clear everyday language  
- ✅ Focused on what and why, not low-level code details  
- ✅ Ready to hand off to SPARC for deep research, spec writing, architecture, coding, testing, and deployment!  
