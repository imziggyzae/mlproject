import os
import sys
from pathlib import Path
import logging as std_logging

from flask import Flask, render_template, request
from jinja2 import FileSystemLoader

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
    return render_template('home.html', error=f"Application error: {str(error)}\n\nDetails logged. Check server logs."), 500


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

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

        return render_template(
            'home.html',
            results=f"Predicted addiction level: {predicted_value}",
            predicted_value=predicted_value,
        )
    except Exception as e:
        app.logger.exception("Prediction request failed")
        import traceback
        error_details = traceback.format_exc()
        app.logger.error(f"Error traceback:\n{error_details}")
        return render_template('home.html', error=f"Prediction error: {str(e)}")


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
