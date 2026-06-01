import os
import sys
from pathlib import Path
import logging as std_logging

from flask import Flask, render_template, render_template_string, request
from jinja2 import FileSystemLoader, TemplateNotFound

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure basic logging early
std_logging.basicConfig(
    level=std_logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

try:
    from src.pipeline.predict_pipeline import CustomData, PredictPipeline
except Exception as e:
    std_logging.error(f"Failed to import predict_pipeline: {e}", exc_info=True)
    raise

# Create Flask app with explicit template folder
template_folder = project_root / 'templates'
application = Flask(__name__, template_folder=str(template_folder))
app = application
app.jinja_loader = FileSystemLoader([str(template_folder)])

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Social Media Addiction Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; line-height: 1.5; }
        .card { max-width: 700px; margin: 0 auto; padding: 1.5rem; border: 1px solid #ccc; border-radius: 10px; }
        a { display: inline-block; margin-top: 1rem; padding: 0.6rem 1rem; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
    </style>
</head>
<body>
    <div class=\"card\">
        <h1>Teen Social Media Addiction Prediction</h1>
        <p>Use this application to estimate the addiction level from a set of behavioral and mental health inputs.</p>
        <a href=\"/predictdata\">Go to prediction form</a>
    </div>
</body>
</html>"""

HOME_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Predict Addiction Level</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2rem; background: #f8f9fa; }
        .container { max-width: 900px; margin: 0 auto; }
        form { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; background: white; padding: 1.5rem; border-radius: 10px; border: 1px solid #ddd; }
        label { display: flex; flex-direction: column; font-weight: 600; gap: 0.4rem; }
        input, select { padding: 0.6rem; border-radius: 5px; border: 1px solid #bbb; }
        button { grid-column: 1 / span 2; padding: 0.8rem; background: #28a745; color: white; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; }
        .result { margin-top: 1rem; background: #fff; padding: 1rem; border-radius: 8px; border: 1px solid #ddd; }
        .error { color: #b30000; background: #ffe5e5; padding: 1rem; border-radius: 8px; }
    </style>
</head>
<body>
    <div class=\"container\">
        <h1>Teen Social Media Addiction Predictor</h1>
        <p>Enter the behavioral and mental health indicators below for a prediction.</p>
        <form method=\"POST\" action=\"/predictdata\">
            <label>Age
                <input type=\"number\" step=\"1\" name=\"age\" required />
            </label>
            <label>Gender
                <select name=\"gender\" required>
                    <option value=\"male\">male</option>
                    <option value=\"female\">female</option>
                </select>
            </label>
            <label>Daily social media hours
                <input type=\"number\" step=\"0.1\" name=\"daily_social_media_hours\" required />
            </label>
            <label>Platform usage
                <select name=\"platform_usage\" required>
                    <option value=\"Instagram\">Instagram</option>
                    <option value=\"TikTok\">TikTok</option>
                    <option value=\"Both\">Both</option>
                </select>
            </label>
            <label>Sleep hours
                <input type=\"number\" step=\"0.1\" name=\"sleep_hours\" required />
            </label>
            <label>Screen time before sleep
                <input type=\"number\" step=\"0.1\" name=\"screen_time_before_sleep\" required />
            </label>
            <label>Academic performance
                <input type=\"number\" step=\"0.1\" name=\"academic_performance\" required />
            </label>
            <label>Physical activity
                <input type=\"number\" step=\"0.1\" name=\"physical_activity\" required />
            </label>
            <label>Social interaction level
                <select name=\"social_interaction_level\" required>
                    <option value=\"low\">low</option>
                    <option value=\"medium\">medium</option>
                    <option value=\"high\">high</option>
                </select>
            </label>
            <label>Stress level
                <input type=\"number\" step=\"0.1\" name=\"stress_level\" required />
            </label>
            <label>Anxiety level
                <input type=\"number\" step=\"0.1\" name=\"anxiety_level\" required />
            </label>
            <label>Depression label
                <input type=\"number\" step=\"0.1\" name=\"depression_label\" required />
            </label>
            <button type=\"submit\">Predict</button>
        </form>
        {% if error %}
            <div class=\"error\">{{ error }}</div>
        {% endif %}
        {% if results %}
            <div class=\"result\">{{ results }}</div>
        {% endif %}
    </div>
</body>
</html>"""


def render_template_fallback(template_name, **context):
    try:
        return render_template(template_name, **context)
    except TemplateNotFound:
        if template_name == 'index.html':
            return render_template_string(INDEX_TEMPLATE, **context)
        if template_name == 'home.html':
            return render_template_string(HOME_TEMPLATE, **context)
        raise

# Log app startup info
app.logger.info(f"Flask app initialized. Project root: {project_root}")
app.logger.info(f"Template folder: {template_folder}")
app.logger.info(f"Template folder exists: {template_folder.exists()}")
app.logger.info(f"Jinja loader searchpath: {app.jinja_loader.searchpath}")


@app.errorhandler(Exception)
def handle_exception(error):
    app.logger.exception("Unhandled application error")
    import traceback
    error_details = traceback.format_exc()
    app.logger.error(f"Error traceback:\n{error_details}")
    return render_template_fallback('home.html', error=f"Application error: {str(error)}\n\nDetails logged. Check server logs."), 500


@app.route('/')
def index():
    return render_template_fallback('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template_fallback('home.html')

    try:
        app.logger.info("Prediction request received")
        
        # Log all form data
        form_data = {
            'age': request.form.get('age'),
            'gender': request.form.get('gender'),
            'daily_social_media_hours': request.form.get('daily_social_media_hours'),
            'platform_usage': request.form.get('platform_usage'),
            'sleep_hours': request.form.get('sleep_hours'),
            'screen_time_before_sleep': request.form.get('screen_time_before_sleep'),
            'academic_performance': request.form.get('academic_performance'),
            'physical_activity': request.form.get('physical_activity'),
            'social_interaction_level': request.form.get('social_interaction_level'),
            'stress_level': request.form.get('stress_level'),
            'anxiety_level': request.form.get('anxiety_level'),
            'depression_label': request.form.get('depression_label'),
        }
        app.logger.info(f"Form data: {form_data}")
        
        data = CustomData(
            age=float(request.form.get('age')),
            gender=request.form.get('gender'),
            daily_social_media_hours=float(request.form.get('daily_social_media_hours')),
            platform_usage=request.form.get('platform_usage'),
            sleep_hours=float(request.form.get('sleep_hours')),
            screen_time_before_sleep=float(request.form.get('screen_time_before_sleep')),
            academic_performance=float(request.form.get('academic_performance')),
            physical_activity=float(request.form.get('physical_activity')),
            social_interaction_level=request.form.get('social_interaction_level'),
            stress_level=float(request.form.get('stress_level')),
            anxiety_level=float(request.form.get('anxiety_level')),
            depression_label=float(request.form.get('depression_label')),
        )
        app.logger.info("CustomData object created")

        pred_df = data.get_data_as_data_frame()
        app.logger.info("Data converted to dataframe")
        
        app.logger.info("Initializing PredictPipeline")
        pipeline = PredictPipeline()
        app.logger.info("Making prediction")
        
        predictions = pipeline.predict(pred_df)
        predicted_value = float(predictions[0])
        app.logger.info(f"Prediction successful: {predicted_value}")

        return render_template_fallback(
            'home.html',
            results=f"Predicted addiction level: {predicted_value}",
            predicted_value=predicted_value,
        )
    except Exception as e:
        app.logger.exception("Prediction request failed")
        import traceback
        error_details = traceback.format_exc()
        app.logger.error(f"Error traceback:\n{error_details}")
        return render_template_fallback('home.html', error=f"Prediction error: {str(e)}")


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
