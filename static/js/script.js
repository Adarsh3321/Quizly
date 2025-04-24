function generateMCQs() {
    const text = document.getElementById("inputText").value;
    const numQuestions = document.getElementById("numQuestions").value;
    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "Generating questions...";

    fetch("/generate_mcqs", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text, num_questions: parseInt(numQuestions) })
    })
    .then(response => response.json())
    .then(data => {
        resultsDiv.innerHTML = "";

        data.mcqs.forEach((mcq, index) => {
            const card = document.createElement("div");
            card.className = "question-card";
            card.innerHTML = `
                <h3>Q${index + 1}: ${mcq.question}</h3>
                <ul>
                    ${mcq.options.map((opt, i) =>
                        `<li ${i === mcq.correct_index ? 'style="font-weight:bold;"' : ''}>${opt}</li>`
                    ).join('')}
                </ul>
                <p><strong>Answer:</strong> ${mcq.answer}</p>
                <small><em>From: ${mcq.source_sentence}</em></small>
            `;
            resultsDiv.appendChild(card);
        });

        const analysis = data.analysis;
        const summary = document.createElement("div");
        summary.innerHTML = `
            <h2>Analysis</h2>
            <p><strong>Total Questions:</strong> ${analysis.total_questions}</p>
            <p><strong>Question Types:</strong> ${JSON.stringify(analysis.question_types)}</p>
            <p><strong>Difficulty (avg):</strong> ${analysis.difficulty_estimate.average.toFixed(2)}</p>
            <p><strong>Coverage Score:</strong> ${analysis.coverage_score}</p>
        `;
        resultsDiv.appendChild(summary);
    })
    .catch(error => {
        resultsDiv.innerHTML = "Error generating MCQs. Check the console.";
        console.error("Error:", error);
    });
}
