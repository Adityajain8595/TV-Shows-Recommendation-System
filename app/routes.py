from flask import Blueprint, render_template, request, jsonify, redirect, url_for
from app.models import recommendation_model
from app.utils import get_poster_url, format_genres

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """Home page with search and filters"""
    return render_template('index.html')

@bp.route('/api/popular-shows')
def popular_shows():
    """Get popular shows with real data from dataframe"""
    popular = recommendation_model.shows.nlargest(12, 'vote_count')[['name', 'poster_path', 'vote_average', 'vote_count', 'number_of_seasons']]
    
    results = []
    for _, show in popular.iterrows():
        results.append({
            'name': show['name'],
            'poster_url': get_poster_url(show['poster_path'], size='w300'),
            'rating': float(show['vote_average']),
            'votes': int(show['vote_count']),
            'seasons': int(show['number_of_seasons'])
        })
    
    return jsonify({'results': results})

@bp.route('/api/search')
def search():
    """API endpoint for searching shows"""
    query = request.args.get('q', '').strip()
    
    if len(query) < 2:
        return jsonify({'results': []})
    
    results = recommendation_model.search_shows(query, limit=15)
    
    formatted_results = []
    for result in results:
        poster_url = get_poster_url(result.get('poster_path'), size='w92')
        
        overview = result.get('overview', '')
        if len(overview) > 100:
            overview = overview[:100] + '...'
        
        formatted_results.append({
            'id': result.get('id'),
            'name': result.get('name'),
            'overview_preview': overview,
            'genres_str': format_genres(result.get('genres', [])),
            'poster_url': poster_url,
            'vote_average': result.get('vote_average', 0),
            'vote_count': result.get('vote_count', 0),
            'number_of_seasons': result.get('number_of_seasons', 0),
            'number_of_episodes': result.get('number_of_episodes', 0)
        })
    
    return jsonify({'results': formatted_results})

@bp.route('/api/stats')
def get_stats():
    """Get min/max values for sliders from dataset"""
    shows = recommendation_model.shows
    
    stats = {
        'min_vote_count': int(shows['vote_count'].min()),
        'max_vote_count': int(shows['vote_count'].max()),
        'min_vote_average': float(shows['vote_average'].min()),
        'max_vote_average': float(shows['vote_average'].max()),
        'min_seasons': int(shows['number_of_seasons'].min()),
        'max_seasons': int(shows['number_of_seasons'].max()),
        'min_episodes': int(shows['number_of_episodes'].min()),
        'max_episodes': int(shows['number_of_episodes'].max())
    }
    
    return jsonify(stats)

@bp.route('/recommendations', methods=['GET'])
def recommendations_page():
    """Main recommendations page with search and filters"""
    show_name = request.args.get('show', '')
    vote_count = int(request.args.get('vote_count', 0))
    vote_average = float(request.args.get('vote_average', 0))
    max_seasons = int(request.args.get('max_seasons', 20))
    max_episodes = int(request.args.get('max_episodes', 500))
    
    result = None
    if show_name:
        result = recommendation_model.get_similar_shows(
            show_name=show_name,
            vote_count=vote_count,
            vote_average=vote_average,
            max_seasons=max_seasons,
            max_episodes=max_episodes,
            top_n=6
        )
        
        if result:
            # Add poster URLs and format
            result['target_show']['poster_url'] = get_poster_url(result['target_show'].get('poster_path'), size='w300')
            result['target_show']['genres_str'] = format_genres(result['target_show'].get('genres', []))
            
            for rec in result['recommendations']:
                rec['poster_url'] = get_poster_url(rec.get('poster_path'), size='w300')
                rec['genres_str'] = format_genres(rec.get('genres', []))
    
    return render_template('recommendations_page.html', 
                         result=result, 
                         search_show=show_name,
                         vote_count=vote_count,
                         vote_average=vote_average,
                         max_seasons=max_seasons,
                         max_episodes=max_episodes)

@bp.route('/recommend', methods=['GET'])
def recommend_redirect():
    """Redirect old recommend route to new recommendations page"""
    show_name = request.args.get('show', '')
    if show_name:
        return redirect(url_for('main.recommendations_page', show=show_name))
    return redirect(url_for('main.recommendations_page'))

@bp.route('/api/recommend')
def api_recommend():
    """REST API endpoint for recommendations"""
    show_name = request.args.get('show_name', '')
    vote_count = int(request.args.get('vote_count', 50))
    vote_average = float(request.args.get('vote_average', 5.0))
    max_seasons = int(request.args.get('max_seasons', 20))
    max_episodes = int(request.args.get('max_episodes', 500))
    
    if not show_name:
        return jsonify({'error': 'show_name parameter is required'}), 400
    
    result = recommendation_model.get_similar_shows(
        show_name=show_name,
        vote_count=vote_count,
        vote_average=vote_average,
        max_seasons=max_seasons,
        max_episodes=max_episodes,
        top_n=6
    )
    
    if result is None:
        return jsonify({'error': f'Show "{show_name}" not found'}), 404
    
    # Add poster URLs
    if result.get('target_show'):
        result['target_show']['poster_url'] = get_poster_url(result['target_show'].get('poster_path'))
    
    for rec in result.get('recommendations', []):
        rec['poster_url'] = get_poster_url(rec.get('poster_path'))
    
    return jsonify(result)